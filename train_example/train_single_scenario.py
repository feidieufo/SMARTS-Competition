import argparse
from pathlib import Path

import ray
from ray import tune
from utils.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from utils.callback import (
    on_episode_start,
    on_episode_step,
    on_episode_end,
)

RUN_NAME = Path(__file__).stem
EXPERIMENT_NAME = "{scenario}-{algorithm}-{n_agent}"


def parse_args():
    parser = argparse.ArgumentParser("train on single scenario")

    # env setting
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name")
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )

    # training setting
    parser.add_argument(
        "--algorithm", type=str, default="PPO", help="training algorithms",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="rllib num workers")
    parser.add_argument(
        "--horizon", type=int, default=1000, help="horizon for a episode"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="Resume training or not."
    )
    parser.add_argument(
        "--restore",
        default=None,
        type=str,
        help="path to restore checkpoint, absolute dir",
    )
    parser.add_argument(
        "--log_dir",
        default="~/smarts/results",
        type=str,
        help="path to store rllib log and checkpoints",
    )
    parser.add_argument("--address", type=str)

    return parser.parse_args()


def main(args):
    # ====================================
    # init env config
    # ====================================

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

    scenario_path = Path(args.scenario).absolute()

    env_config = {
        "seed": 42,
        "scenarios": [str(scenario_path)],
        "headless": args.headless,
        "agent_specs": agent_specs,
    }

    # ====================================
    # init tune config
    # ====================================
    tune_config = {
        "env": RLlibHiWayEnv,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "default_policy": (None, OBSERVATION_SPACE, ACTION_SPACE, {},)
            },
            "policy_mapping_fn": lambda agent_id: "default_policy",
        },
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
        # "batch_mode": "complete_episodes",
        # "use_gae": False,
    }

    if args.algorithm == "PPO":
        tune_config.update(
            {
                "lambda": 0.95,
                "clip_param": 0.2,
                "num_sgd_iter": 10,
                "sgd_minibatch_size": 1024,
            }
        )
    elif args.algorithm in ["A2C", "A3C"]:
        tune_config.update(
            {"lambda": 0.95,}
        )

    # ====================================
    # init log and checkpoint dir_info
    # ====================================
    experiment_name = EXPERIMENT_NAME.format(
        scenario=scenario_path.stem, algorithm=args.algorithm, n_agent=1,
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
        args.algorithm,
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
    )

    print(analysis.dataframe().head())


if __name__ == "__main__":
    args = parse_args()
    main(args)
