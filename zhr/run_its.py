from pathlib import Path

import gym

from agent import agent_spec

scenario_paths = [(
    # Path(__file__).parent / "../dataset/intersection_4lane_sv"
# ).resolve(), (
    # Path(__file__).parent / "../dataset/intersection_4lane_sv_up"
# ).resolve(), (
    Path(__file__).parent / "../dataset/intersection_4lane_sv_right"
).resolve()]

# scenario_paths = [(
    # Path(__file__).parent / "../dataset/test/intersection_4lane_sv"
# ).resolve(),(
#     Path(__file__).parent / "../dataset/test/intersection_4lane_sv_up"
# ).resolve()]


AGENT_ID = "Agent-007"


def main():

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenario_paths,
        agent_specs={AGENT_ID: agent_spec},
        # set headless to false if u want to use envision
        headless=False,
        visdom=False,
        seed=42,
    )

    agent = agent_spec.build_agent()

    # while True:
    #     step = 0
    #     observations = env.reset()
    #     total_reward = 0.0
    #     dones = {"__all__": False}

    #     while not dones["__all__"]:
    #         step += 1
    #         agent_obs = observations[AGENT_ID]
    #         agent_action = agent.act(agent_obs)
    #         a1 = agent_action[:-1]
    #         a2 = agent_action[-1]
    #         for i in range(int(a2)+1):
    #             observations, rewards, dones, _ = env.step({AGENT_ID: a1})
    #             total_reward += rewards[AGENT_ID]
    #             if dones["__all__"]:
    #                 break
            
    #         print("step ", step, "a1:", a1, "a2:", int(a2)+1)
    #         if dones["__all__"]:
    #             break            
    #     print("Accumulated reward:", total_reward)

    while True:
        step = 0
        observations = env.reset()
        total_reward = 0.0
        dones = {"__all__": False}

        while not dones["__all__"]:
            step += 1
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, _ = env.step({AGENT_ID: agent_action})
            total_reward += rewards[AGENT_ID]
            print("step ", step, "a ", agent_action)   
            if dones["__all__"]:
                break
            
        print("Accumulated reward:", total_reward)

    env.close()


if __name__ == "__main__":
    main()
