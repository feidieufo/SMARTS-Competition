import argparse
import ray
import collections
import gym

from pathlib import Path

from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.rollout import default_policy_agent_mapping, DefaultMapping
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.scenario import Scenario

from benchmark.agents import load_config
from benchmark.metrics import basic_metrics as metrics
from ray.rllib.models import ModelCatalog
from utils.continuous_space_a3_36_head_benchmark import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
from smarts.env.rllib_hiway_env import RLlibHiWayEnv

def parse_args():
    parser = argparse.ArgumentParser("Run evaluation")
    parser.add_argument(
        "--scenario", type=str, help="Scenario name",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument(
        "--paradigm",
        type=str,
        default="decentralized",
        help="Algorithm paradigm, decentralized (default) or centralized",
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )
    parser.add_argument("--config_file", "-f", type=str)
    return parser.parse_args()

scenario_paths = [(
#     Path(__file__).parent / "../dataset_public/all_loop/all_loop_a"
# ).resolve(), (
#     Path(__file__).parent / "../dataset_public/merge_loop/merge_a"
# ).resolve(), (
    # Path(__file__).parent / "../dataset_public/intersection_loop/its_a"
    Path(__file__).parent / "../dataset/intersection_4lane_sv"
# ).resolve(), (
    # Path(__file__).parent / "../dataset/intersection_4lane_sv_up"
#     Path(__file__).parent / "../dataset_public/mixed_loop/its_merge_a"
# ).resolve(), (
#     Path(__file__).parent / "../dataset/intersection_4lane_sv_right"
#     Path(__file__).parent / "../dataset_public/mixed_loop/roundabout_its_a"
# ).resolve(), (
#     Path(__file__).parent / "../dataset_public/mixed_loop/roundabout_merge_a"
).resolve()]
print(f"training on {scenario_paths}")
import importlib
def _get_trainer(path, name):
    module = importlib.import_module(path)
    trainer = module.__getattribute__(name)
    return trainer

def main(
    scenario,
    config_file,
    checkpoint,
    num_steps=1000,
    num_episodes=10,
    paradigm="decentralized",
    headless=False,
):

    class MultiEnv(RLlibHiWayEnv):
        def __init__(self, env_config):
            env_config["scenarios"] = [
                scenario_paths[(env_config.worker_index - 1) % len(scenario_paths)]
            ]
            super(MultiEnv, self).__init__(config=env_config)

    from utils.fc_model import FullyConnectedNetwork
    ModelCatalog.register_custom_model("my_fc", FullyConnectedNetwork)
    agent_spec.info_adapter = metrics.agent_info_adapter
    agent_specs = {"AGENT-007": agent_spec}
    env_config = {
        "seed": 42,
        "scenarios": [scenario_paths],
        "headless": headless,
        "agent_specs": agent_specs,
    }
    tune_config = {
        "env": MultiEnv,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "default_policy": (None, OBSERVATION_SPACE, ACTION_SPACE, {},)
            },
            "policy_mapping_fn": lambda agent_id: "default_policy",
        },
        # "model": {
        #     "custom_model": "my_rnn",
        # },
        "framework": "torch",
        "lr": 1e-4,
        "log_level": "WARN",
        "num_workers": 1,
        "horizon": num_steps,
        "train_batch_size": 10240 * 3,

        # "observation_filter": "MeanStdFilter",
        # "batch_mode": "complete_episodes",
        # "grad_clip": 0.5, 

        # "model":{
        #     "use_lstm": True,
        # },
    }

    ray.init()
    trainer_cls = _get_trainer("ray.rllib.agents.ppo", "PPOTrainer")

    trainer = trainer_cls(env=tune_config["env"], config=tune_config)

    from pathlib import Path
    load_file = Path(__file__).parent / checkpoint
    trainer.restore(str(load_file))
    rollout(trainer, None, num_steps, num_episodes)
    trainer.stop()


def rollout(trainer, env_name, num_steps, num_episodes=0):
    """ Reference: https://github.com/ray-project/ray/blob/master/rllib/rollout.py
    """
    policy_agent_mapping = default_policy_agent_mapping
    if hasattr(trainer, "workers") and isinstance(trainer.workers, WorkerSet):
        env = trainer.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if trainer.workers.local_worker().multiagent:
            policy_agent_mapping = trainer.config["multiagent"]["policy_mapping_fn"]

        policy_map = trainer.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        env = gym.make(env_name)
        multiagent = False
        try:
            policy_map = {DEFAULT_POLICY_ID: trainer.policy}
        except AttributeError:
            raise AttributeError(
                "Agent ({}) does not have a `policy` property! This is needed "
                "for performing (trained) agent rollouts.".format(trainer)
            )
        use_lstm = {DEFAULT_POLICY_ID: False}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    metrics_obj = metrics.Metric(num_episodes)

    for episode in range(num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]]
        )
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]]
        )
        prev_rewards = collections.defaultdict(lambda: 0.0)
        done = False
        reward_total = 0.0
        step = 0
        while not done and step < num_steps:
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id)
                    )
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = trainer.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                        )
                        agent_states[agent_id] = p_state
                    else:
                        a_action = trainer.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                        )
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)

            metrics_obj.log_step(multi_obs, reward, done, info, episode=episode)

            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            # filter dead agents
            if multiagent:
                next_obs = {
                    agent_id: obs
                    for agent_id, obs in next_obs.items()
                    if not done[agent_id]
                }

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward

            step += 1
            obs = next_obs
        print("\nEpisode #{}: steps: {} reward: {}".format(episode, step, reward_total))
        if done:
            episode += 1
    print("\n metrics: {}".format(metrics_obj.compute()))


if __name__ == "__main__":
    args = parse_args()
    main(
        scenario=args.scenario,
        config_file=args.config_file,
        checkpoint=args.checkpoint,
        num_steps=args.num_steps,
        num_episodes=args.num_episodes,
        paradigm=args.paradigm,
        headless=args.headless,
    )
