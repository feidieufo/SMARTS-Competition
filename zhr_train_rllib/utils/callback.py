"""
This file contains some callbacks for RLlib like record some metrics.`
"""
import numpy as np


def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["ego_speed"] = []
    episode.user_data["dis_from_center"] = []

def on_episode_step(info):
    episode = info["episode"]
    agent_speeds = []
    dis_from_center = []
    for id, obs in episode._agent_to_last_raw_obs.items():
        agent_speeds.append(obs["speed"])
        dis_from_center.append(obs["distance_from_center"])
    episode.user_data["ego_speed"].append(np.mean(agent_speeds))
    episode.user_data["dis_from_center"].append(np.mean(dis_from_center))

def on_episode_end(info):
    episode = info["episode"]
    mean_ego_speed = np.mean(episode.user_data["ego_speed"])
    episode.custom_metrics["mean_ego_speed"] = mean_ego_speed
    mean_dis_center = np.mean(episode.user_data["dis_from_center"])
    episode.custom_metrics["dis_from_center"] = mean_dis_center

    # wait for distance as score is ok
    agent_scores = []
    for info in episode._agent_to_last_info.values():
        agent_scores.append(info["score"])
    mean_dis = np.mean(agent_scores)
    episode.custom_metrics["distance_travelled"] = mean_dis
