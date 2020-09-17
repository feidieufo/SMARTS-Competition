import core_multi_branch_torch as core
import numpy as np
import gym
import argparse
import scipy
from scipy import signal
import pickle

import os
from utils.logx import EpochLogger
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.normalization import *
import json
from utils.utils import discount_path, get_path_indices
from utils.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
from pathlib import Path
from ray.rllib.models import ModelCatalog
import time


class ReplayBuffer:
    def __init__(self, size, state_dim, act_dim, gamma=0.99, lam=0.95, is_gae=True):
        self.size = size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.is_gae = is_gae
        self.reset()

    def reset(self):
        self.state = np.zeros((self.size,) + self.state_dim, np.float32)
        if type(self.act_dim) == np.int64 or type(self.act_dim) == np.int:
            self.action = np.zeros((self.size, ), np.int32)
        else:
            self.action = np.zeros((self.size,) + self.act_dim, np.float32)
        self.v = np.zeros((self.size, ), np.float32)
        self.reward = np.zeros((self.size, ), np.float32)
        self.adv = np.zeros((self.size, ), np.float32)
        self.mask = np.zeros((self.size, ), np.float32)
        self.cline = np.zeros((self.size, ), np.long)
        self.ptr, self.path_start = 0, 0

    def add(self, s, a, r, mask, cline):
        if self.ptr < self.size:
            self.state[self.ptr] = s
            self.action[self.ptr] = a
            self.reward[self.ptr] = r
            self.mask[self.ptr] = mask
            self.cline[self.ptr] = cline
            self.ptr += 1

    def update_v(self, v, pos):
        self.v[pos] = v

    def finish_path(self):
        """
          Calculate GAE advantage, discounted returns, and
          true reward (average reward per trajectory)

          GAE: delta_t^V = r_t + discount * V(s_{t+1}) - V(s_t)
          using formula from John Schulman's code:
          V(s_t+1) = {0 if s_t is terminal
                   {v_s_{t+1} if s_t not terminal and t != T (last step)
                   {v_s if s_t not terminal and t == T
          """
        v_ = np.concatenate([self.v[1:], self.v[-1:]], axis=0) * self.mask
        adv = self.reward + self.gamma * v_ - self.v

        indices = get_path_indices(self.mask)

        for (start, end) in indices:
            self.adv[start:end] = discount_path(adv[start:end], self.gamma * self.lam)
            if not self.is_gae:
                self.reward[start:end] = discount_path(self.reward[start:end], self.gamma)
        if self.is_gae:
            self.reward = self.adv + self.v

        self.adv = (self.adv - np.mean(self.adv))/(np.std(self.adv) + 1e-8)

    def get_batch(self, batch=100, shuffle=True):
        if shuffle:
            indices = np.random.permutation(self.size)
        else:
            indices = np.arange(self.size)

        for idx in np.arange(0, self.size, batch):
            pos = indices[idx:(idx + batch)]
            yield (self.state[pos], self.action[pos], self.reward[pos], 
            self.adv[pos], self.v[pos], self.cline[pos])

class StateNormalization:
    def __init__(self, pre_filter, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.pre_filter = pre_filter

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        x = self.pre_filter(x)
        obs = x["obs"]
        if update:
            self.rs.push(obs)
        if self.demean:
            obs = obs - self.rs.mean
        if self.destd:
            obs = obs / (self.rs.std + 1e-8)
        if self.clip:
            obs = np.clip(obs, -self.clip, self.clip)
        x["obs"] = obs
        return x

    def reset(self):
        self.pre_filter.reset()

    @staticmethod
    def output_shape(input_space):
        return input_space.shape

class SmartsProcess:
    def __init__(self, pre_filter):
        self.pre_filter = pre_filter

    def __call__(self, x, update=True):
        x = self.pre_filter(x[AGENT_ID])
        cline = x.pop("cline")
        x = preprocessor.transform(x)
        return {"obs": x, "cline":cline}
    
    def reset(self):
        self.pre_filter.reset()  

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lam', default=0.95, type=float)
    parser.add_argument('--c_en', default=0.01, type=float)    
    parser.add_argument('--c_vf', default=0.5, type=float)
    parser.add_argument('--a_update', default=10, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--log', type=str, default="logs")
    parser.add_argument('--steps', default=300, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--env', default="CartPole-v1")
    parser.add_argument('--env_num', default=4, type=int)
    parser.add_argument('--exp_name', default="ppo_cartpole")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch', default=50, type=int)
    parser.add_argument('--norm_state', action="store_true")
    parser.add_argument('--norm_rewards', default=None, type=str)
    parser.add_argument('--is_clip_v', action="store_true")
    parser.add_argument('--last_v', action="store_true")
    parser.add_argument('--is_gae', action="store_true")
    parser.add_argument('--max_grad_norm', default=-1, type=float)
    parser.add_argument('--anneal_lr', action="store_true")
    parser.add_argument('--debug', action="store_false")
    parser.add_argument('--log_every', default=5, type=int)
    parser.add_argument('--target_kl', default=0.03, type=float)   
    args = parser.parse_args()

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    from utils.utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger = EpochLogger(**logger_kwargs)
    with open(os.path.join(logger.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    writer = SummaryWriter(os.path.join(logger.output_dir, "logs"))

    scenario_root = (Path(__file__).parent / "../dataset_public").resolve()

    scenario_paths = [
        scenario
        for scenario_dir in scenario_root.iterdir()
        for scenario in scenario_dir.iterdir()
        if scenario.is_dir()
    ]

    print(f"training on {scenario_paths}")

    AGENT_ID = "Agent-007"
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenario_paths,
        agent_specs={AGENT_ID: agent_spec},
        # set headless to false if u want to use envision
        headless=False,
        visdom=False,
        seed=args.seed,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    preprocessor = ModelCatalog.get_preprocessor_for_space(OBSERVATION_SPACE)

    state_dim = 0
    for val in OBSERVATION_SPACE.spaces.values():
        state_dim += val.shape[0]
    state_dim = (state_dim, )

    if type(ACTION_SPACE) == gym.spaces.Discrete:
        act_dim = ACTION_SPACE.n
        action_max = 1
    else:
        act_dim = ACTION_SPACE.shape
        action_max = ACTION_SPACE.high[0]
    ppo = core.PPO(state_dim, act_dim, action_max, 0.2, device, lr_a=args.lr, max_grad_norm=args.max_grad_norm,
                   anneal_lr=args.anneal_lr, train_steps=args.iteration, c_en=args.c_en, c_vf=args.c_vf)
    replay = ReplayBuffer(args.steps, state_dim, act_dim, is_gae=args.is_gae)

    state_norm = Identity()
    reward_norm = Identity()
    state_norm = SmartsProcess(state_norm)
    if args.norm_state:
        state_norm = StateNormalization(state_norm, state_dim, clip=10.0)
    if args.norm_rewards == "rewards":
        reward_norm = AutoNormalization(reward_norm, (), clip=10.0)
    elif args.norm_rewards == "returns":
        reward_norm = RewardFilter(reward_norm, (), clip=10.0)

    state_norm.reset()
    reward_norm.reset()
    obs = env.reset()
    obs = state_norm(obs)
    for iter in range(args.iteration):
        start = time.time()
        ppo.train()
        replay.reset()
        rew = 0
        epoch = 0

        for step in range(args.steps):
            state_tensor = torch.tensor(obs["obs"], dtype=torch.float32, device=device).unsqueeze(0)
            c_tensor = torch.tensor(obs["cline"], dtype=torch.long, device=device).unsqueeze(0)
            a_tensor = ppo.actor.select_action(state_tensor, c_tensor)
            a = a_tensor.detach().cpu().numpy()
            obs_, r, done, _ = env.step({AGENT_ID: a})
            rew += r[AGENT_ID]
            r = reward_norm(r[AGENT_ID])
            mask = 1-done[AGENT_ID]

            replay.add(obs["obs"], a, r, mask, obs["cline"])

            obs = obs_
            if done[AGENT_ID] or step == args.steps-1:
                logger.store(reward=rew)
                if done[AGENT_ID]:
                    rew = 0
                    obs = env.reset()
                    state_norm.reset()
                    reward_norm.reset()
                    epoch += 1
            obs = state_norm(obs)

        state = replay.state
        for idx in np.arange(0, state.shape[0], args.batch):
            if idx + args.batch <= state.shape[0]:
                pos = np.arange(idx, idx + args.batch)
            else:
                pos = np.arange(idx, state.shape[0])
            s = torch.tensor(state[pos], dtype=torch.float32).to(device)
            c = torch.tensor(replay.cline[pos], dtype=torch.long).to(device)
            v = ppo.getV(s, c).detach().cpu().numpy()
            replay.update_v(v, pos)
        replay.finish_path()

        ppo.update_a()
        for i in range(args.a_update):
            for (s, a, r, adv, v, c) in replay.get_batch(batch=args.batch):
                s_tensor = torch.tensor(s, dtype=torch.float32, device=device)
                a_tensor = torch.tensor(a, dtype=torch.float32, device=device)
                adv_tensor = torch.tensor(adv, dtype=torch.float32, device=device)
                r_tensor = torch.tensor(r, dtype=torch.float32, device=device)
                v_tensor = torch.tensor(v, dtype=torch.float32, device=device)
                c_tensor = torch.tensor(c, dtype=torch.long, device=device)

                info = ppo.train_ac(s_tensor, a_tensor, adv_tensor, r_tensor, 
                    v_tensor, c_tensor, is_clip_v=args.is_clip_v)

                if args.debug:
                    logger.store(aloss=info["aloss"])
                    logger.store(vloss=info["vloss"])
                    logger.store(entropy=info["entropy"])
                    logger.store(kl=info["kl"])
            
            if logger.get_stats("kl", with_min_and_max=True)[3] > args.target_kl:
                print("stop at:", str(i))
                break

        if args.anneal_lr:
            ppo.lr_scheduler()
        
        writer.add_scalar("reward", logger.get_stats("reward")[0], global_step=iter)
        writer.add_histogram("action", np.array(replay.action), global_step=iter)
        if args.debug:
            writer.add_scalar("aloss", logger.get_stats("aloss")[0], global_step=iter)
            writer.add_scalar("vloss", logger.get_stats("vloss")[0], global_step=iter)
            writer.add_scalar("entropy", logger.get_stats("entropy")[0], global_step=iter)
            writer.add_scalar("kl", logger.get_stats("kl")[0], global_step=iter)              

        end = time.time()
        logger.log_tabular('Epoch', iter)
        logger.log_tabular('Episode', epoch)
        logger.log_tabular('time', (end-start)/60)
        logger.log_tabular("reward", with_min_and_max=True)
        if args.debug:
            logger.log_tabular("aloss", with_min_and_max=True)
            logger.log_tabular("vloss", with_min_and_max=True)
            logger.log_tabular("entropy", with_min_and_max=True)
            logger.log_tabular("kl", with_min_and_max=True)
        logger.dump_tabular()

        if not os.path.exists(os.path.join(logger.output_dir, "checkpoints")):
            os.makedirs(os.path.join(logger.output_dir, "checkpoints"))
        if iter % args.log_every == 0:
            state = {
                "iter": iter,
                "actor": ppo.actor.state_dict(),
                "critic": ppo.critic.state_dict(),
                "opti": ppo.opti.state_dict(),
            }
            torch.save(state, os.path.join(logger.output_dir, "checkpoints", str(iter) + '.pth'))
            norm = {"state": state_norm, "reward": reward_norm}
            with open(os.path.join(logger.output_dir, "checkpoints", str(iter) + '.pkl'), "wb") as f:
                pickle.dump(norm, f)


