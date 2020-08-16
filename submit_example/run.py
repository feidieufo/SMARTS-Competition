from pathlib import Path

import gym

from agent import agent_spec

# Path to the scenario to test
scenario_path = (
    Path(__file__).parent / "../../dataset_public/roundabout_loop/roundabout_a"
).resolve()

AGENT_ID = "Agent-007"


def main():

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario_path],
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
            print("step ", step)
            step += 1
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, _ = env.step({AGENT_ID: agent_action})
            total_reward += rewards[AGENT_ID]
        print("Accumulated reward:", total_reward)

    env.close()


if __name__ == "__main__":
    main()
