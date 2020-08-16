import argparse
from pathlib import Path

import gym

from utils.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
from utils.saved_model import RLlibTFCheckpointPolicy


def parse_args():
    parser = argparse.ArgumentParser("run simple keep lane agent")
    # env setting
    parser.add_argument("--scenario", "-s", type=str, help="Path to scenario")
    parser.add_argument("--load_path", "-p", type=str, help="path to stored model")
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )

    return parser.parse_args()


def main(args):
    # scenario_path = Path(args.scenario).absolute()

    AGENT_ID = "AGENT-007"

    agent_spec.policy_builder = lambda: RLlibTFCheckpointPolicy(
        Path(args.load_path).absolute(),
        "PPO",
        "default_policy",
        OBSERVATION_SPACE,
        ACTION_SPACE,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[Path(args.scenario).absolute()],
        agent_specs={AGENT_ID: agent_spec},
        # set headless to false if u want to use envision
        headless=False,
        visdom=False,
        seed=42,
    )

    agent = agent_spec.build_agent()

    while True:
        step = 0
        observations = env.reset()
        total_reward = 0.0
        dones = {"__all__": False}

        while not dones["__all__"]:
            step += 1
            print(step)
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, _ = env.step({AGENT_ID: agent_action})
            total_reward += rewards[AGENT_ID]
        print("Accumulated reward:", total_reward)

    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
