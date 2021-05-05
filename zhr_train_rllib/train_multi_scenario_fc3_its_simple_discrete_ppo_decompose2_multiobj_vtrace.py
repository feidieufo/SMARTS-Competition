import argparse
from pathlib import Path

import ray
from ray import tune
from zhr_train_rllib.utils.discrete_space_36_head_benchmark_ogm_9_decompose2_multiobj import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
from zhr_train_rllib.ppo_policy_modeldist_multiv_multiobj_vtrace import PPOTorchPolicy
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
    # Path(__file__).parent / "../dataset_public/intersection_loop/its_a"
    # Path(__file__).parent / "../dataset/intersection_4lane_sv"
# ).resolve(), (
    # Path(__file__).parent / "../dataset/intersection_4lane_sv_up"
#     Path(__file__).parent / "../dataset_public/mixed_loop/its_merge_a"
# ).resolve(), (
#     Path(__file__).parent / "../dataset/intersection_4lane_sv_right"
#     Path(__file__).parent / "../dataset_public/mixed_loop/roundabout_its_a"
# ).resolve(), (
#     Path(__file__).parent / "../dataset_public/mixed_loop/roundabout_merge_a"
    Path(__file__).parent / "../dataset/simple"
).resolve()]
print(f"training on {scenario_paths}")

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
    parser.add_argument("--exper", type=str, default="multi_scenarios")
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
        ray.init(webui_host="127.0.0.1")
    else:
        ray.init(local_mode=True, webui_host="127.0.0.1")
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
            # "fcnet_hiddens": [256, 256],
            # "fcnet_activation": "relu",
        #     "vf_share_layers": True,
        # },
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
        "train_batch_size": 10240 * 3,

        # "grad_clip": 0.5,

        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": "zhr_train_rllib.stochastic_sampling_multiobj_vtrace.StochasticSampling",
            # Add constructor kwargs here (if any).
            "initial_epsilon": 1.0,
            "final_epsilon": 0.5,
            "epsilon_timesteps": 200000,  # Timesteps over which to anneal epsilon.
        },


        # "observation_filter": "MeanStdFilter",

        # "model":{
        #     "use_lstm": True,
        # },
    }

    tune_config.update(
        {
            "lambda": 0.95,
            "clip_param": 0.2,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 1024,
            "gamma": 0.995,
        }
    )

    num_samples = 4
    if not args.no_debug:
        tune_config.update(
            {
                "train_batch_size": 102 * 3,
                "sgd_minibatch_size": 10,
            }
        )
        num_samples = 1

    # ====================================
    # init log and checkpoint dir_info
    # ====================================
    experiment_name = EXPERIMENT_NAME.format(
        scenario=args.exper, algorithm="PPO", n_agent=1,
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
        name=experiment_name,
        stop={"timesteps_total": 10000000/2},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir=str(log_dir),
        resume=args.resume,
        restore=restore_path,
        max_failures=1000,
        export_formats=["model", "checkpoint"],
        config=tune_config,
        num_samples=num_samples,
    )

    print(analysis.dataframe().head())


if __name__ == "__main__":
    args = parse_args()
    main(args)