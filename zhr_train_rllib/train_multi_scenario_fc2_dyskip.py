import argparse
from pathlib import Path

import ray
from ray import tune
from zhr_train_rllib.utils.continuous_space_a2_36_full_intersec_2 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
from zhr_train_rllib.ppo_policy_dyskip import PPOTorchPolicy
from .fc_model import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog

from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from .utils.callback import (
    on_episode_start,
    on_episode_step,
    on_episode_end,
)

RUN_NAME = Path(__file__).stem
EXPERIMENT_NAME = "{scenario}-{algorithm}-{n_agent}"

scenario_root = (Path(__file__).parent / "../dataset_public").resolve()

scenario_paths = [
    scenario
    for scenario_dir in scenario_root.iterdir()
    for scenario in scenario_dir.iterdir()
    if scenario.is_dir()
]

print(f"training on {scenario_paths}")
scenario_paths = [(
#     Path(__file__).parent / "../dataset_public/all_loop/all_loop_a"
# ).resolve(), (
#     Path(__file__).parent / "../dataset_public/merge_loop/merge_a"
# ).resolve(), (
    Path(__file__).parent / "../dataset_public/intersection_loop/its_a"
# ).resolve(), (
#     Path(__file__).parent / "../dataset_public/mixed_loop/its_merge_a"
# ).resolve(), (
#     Path(__file__).parent / "../dataset_public/mixed_loop/roundabout_its_a"
# ).resolve(), (
#     Path(__file__).parent / "../dataset_public/mixed_loop/roundabout_merge_a"
# ).resolve(), (
    # Path(__file__).parent / "../dataset_public/simple_loop/simpleloop_a"
).resolve()]
print(f"training on {scenario_paths}")

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
import numpy as np
import torch
class TorchDyDistribution(TorchDistributionWrapper):
    @override(ActionDistribution)
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        mean, log_std = torch.chunk(self.inputs[:,:-4], 2, dim=1)
        dy_num = self.inputs[:,-4:]
        self.dist1 = torch.distributions.normal.Normal(mean, torch.exp(log_std))
        self.dist2 = torch.distributions.categorical.Categorical(
            logits=dy_num)

    @override(ActionDistribution)
    def deterministic_sample(self):
        mean = self.dist1.mean
        dy = self.dist2.probs.argmax(dim=1).unsqueeze(0)
        self.last_sample = torch.cat([mean, dy.float()], dim=1)
        return self.last_sample

    @override(ActionDistribution)
    def sample(self):
        mean = self.dist1.sample()
        dy = self.dist2.sample().unsqueeze(0)
        self.last_sample = torch.cat([mean, dy.float()], dim=1)
        return self.last_sample

    @override(ActionDistribution)
    def sampled_action_logp(self):
        assert self.last_sample is not None
        return self.logp(self.last_sample)

    @override(TorchDistributionWrapper)
    def logp(self, actions):
        a1 = actions[:, :-1]
        a2 = actions[:, -1]
        l1 = self.dist1.log_prob(a1)
        l2 = self.dist2.log_prob(a2)
        return self.dist1.log_prob(a1).sum(-1) + self.dist2.log_prob(a2)

    @override(TorchDistributionWrapper)
    def entropy(self):
        e1 = self.dist1.entropy()
        e2 = self.dist2.entropy()
        return self.dist1.entropy().sum(-1) + self.dist2.entropy()  

    @override(ActionDistribution)
    def kl(self, other):
        return torch.distributions.kl.kl_divergence(self.dist1, other.dist1).sum(-1) + \
                torch.distributions.kl.kl_divergence(self.dist2, other.dist2)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return (np.prod(action_space.shape)-1) * 2 + 4


from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, execution_plan, validate_config
PPOTrainer = build_trainer(
    name="PPO_TORCH",
    default_config=DEFAULT_CONFIG,
    default_policy=PPOTorchPolicy,
    execution_plan=execution_plan,
    validate_config=validate_config)

def parse_args():
    parser = argparse.ArgumentParser("train on multi scenarios")

    # env setting
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name")
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )


    parser.add_argument("--num_workers", type=int, default=1, help="rllib num workers")
    parser.add_argument(
        "--horizon", type=int, default=1000, help="horizon for a episode"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="Resume training or not."
    )
    parser.add_argument(
        "--no_debug", default=False, action="store_true"
    )
    parser.add_argument(
        "--restore",
        default=None,
        type=str,
        help="path to restore checkpoint, absolute dir",
    )
    parser.add_argument(
        "--log_dir",
        default="~/workspace/results",
        type=str,
        help="path to store rllib log and checkpoints",
    )

    parser.add_argument("--address", type=str)

    return parser.parse_args()


def main(args):
    # ====================================
    # init env config
    # ====================================
    if args.no_debug:
        ray.init()
    else:
        ray.init(local_mode=True)
    # use ray cluster for training
    # ray.init(
    #     address="auto" if args.address is None else args.address,
    #     redis_password="5241590000000000",
    # )
    #
    # print(
    #     "--------------- Ray startup ------------\n{}".format(
    #         ray.state.cluster_resources()
    #     )
    # )

    agent_specs = {"AGENT-007": agent_spec}

    env_config = {
        "seed": 42,
        "scenarios": [scenario_paths],
        "headless": args.headless,
        "agent_specs": agent_specs,
    }

    # ====================================
    # init tune config
    # ====================================
    class MultiEnv(RLlibHiWayEnv):
        def __init__(self, env_config):
            env_config["scenarios"] = [
                scenario_paths[(env_config.worker_index - 1) % len(scenario_paths)]
            ]
            super(MultiEnv, self).__init__(config=env_config)

        def step(self, agent_actions):
            total_reward = 0.0
            rewards = {
                agent_id: 0.0
                for agent_id in agent_actions
            }

            for agent_id in agent_actions:
                skip = agent_actions[agent_id][-1]
            agent_actions = {agent_id: actions[:-1] for agent_id, actions in agent_actions.items()}  

            for i in range(int(skip)+1):
                obs, r, done, info = super().step(agent_actions)

                for agent_id in agent_actions:
                    rewards[agent_id] += r[agent_id]

                if done["__all__"]:
                    break

            return obs, rewards, done, info

    ModelCatalog.register_custom_model("my_fc", FullyConnectedNetwork)
    ModelCatalog.register_custom_action_dist("my_dist", TorchDyDistribution)
    tune_config = {
        "env": MultiEnv,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "default_policy": (None, OBSERVATION_SPACE, ACTION_SPACE, {},)
            },
            "policy_mapping_fn": lambda agent_id: "default_policy",
        },
        "model": {
            "custom_model": "my_fc",
            "custom_action_dist": "my_dist",
        },
        "framework": "torch",
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
        },
        "lr": 1e-4,
        "log_level": "WARN",
        "num_workers": args.num_workers,
        "horizon": args.horizon,
        "train_batch_size": 5120*3,

        # "observation_filter": "MeanStdFilter",
        # "batch_mode": "complete_episodes",
        # "grad_clip": 0.5, 

        # "model":{
        #     "use_lstm": True,
        # },
    }

    tune_config.update(
        {
            "lambda": 0.95,
            "clip_param": 0.2,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 512,

            "gamma": 0.995,
        }
    )

    # ====================================
    # init log and checkpoint dir_info
    # ====================================
    experiment_name = EXPERIMENT_NAME.format(
        scenario="multi_scenarios_test", algorithm="PPO", n_agent=1,
    )

    log_dir = Path(args.log_dir).expanduser().absolute() / RUN_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpointing at {log_dir}")

    if args.restore:
        restore_path = Path(args.restore).expanduser()
        print(f"Loading model from {restore_path}")
    else:
        restore_path = None

    # run experiments
    analysis = tune.run(
        PPOTrainer,
        # "PPO",
        name=experiment_name,
        stop={"time_total_s": 24 * 60 * 60},
        checkpoint_freq=2,
        checkpoint_at_end=True,
        local_dir=str(log_dir),
        resume=args.resume,
        restore=restore_path,
        max_failures=1000,
        export_formats=["model", "checkpoint"],
        config=tune_config,
        num_samples=4, 
    )

    print(analysis.dataframe().head())


if __name__ == "__main__":
    args = parse_args()
    main(args)
