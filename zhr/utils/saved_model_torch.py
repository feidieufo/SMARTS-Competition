"""
This file contains RLlib policy reload for evaluation usage, not for training.
"""
import os
import pickle

import gym
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from smarts.core.agent import AgentPolicy

import torch

class RLlibTorchGRUPolicy(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, observation_space, action_space):
        self._checkpoint_path = load_path
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        flat_obs_space = self._prep.observation_space

        ray.init(ignore_reinit_error=True, local_mode=True)

        from utils.ppo_policy import PPOTorchPolicy as LoadPolicy
        from utils.rnn_model import RNNModel
        ModelCatalog.register_custom_model("my_rnn", RNNModel)
        config = ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG.copy()
        config['num_workers'] = 0
        config["model"]["custom_model"] = "my_rnn"

        self.policy = LoadPolicy(flat_obs_space, self._action_space, config)
        objs = pickle.load(open(self._checkpoint_path, "rb"))
        objs = pickle.loads(objs["worker"])
        state = objs["state"]
        filters = objs["filters"]
        self.filters = filters[self._policy_name]
        weights = state[self._policy_name]
        weights.pop("_optimizer_variables")
        self.policy.set_weights(weights)
        self.model = self.policy.model

        self.rnn_state = self.model.get_initial_state()
        self.rnn_state = [torch.reshape(self.rnn_state[0], shape=(1, -1))]

    def act(self, obs):

        # single infer
        obs = self._prep.transform(obs)
        obs = self.filters(obs, update=False)
        action, self.rnn_state, _ = self.policy.compute_actions([obs], self.rnn_state, explore=False)
        action = action[0]

        return action

class RLlibTorchGRUDVEPolicy(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, observation_space, action_space):
        self._checkpoint_path = load_path
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        flat_obs_space = self._prep.observation_space

        ray.init(ignore_reinit_error=True, local_mode=True)

        from utils.ppo_policy import PPOTorchPolicy as LoadPolicy
        from utils.rnn_model import RNNDVEModel
        ModelCatalog.register_custom_model("my_rnn", RNNDVEModel)
        config = ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG.copy()
        config['num_workers'] = 0
        config["model"]["custom_model"] = "my_rnn"

        self.policy = LoadPolicy(flat_obs_space, self._action_space, config)
        objs = pickle.load(open(self._checkpoint_path, "rb"))
        objs = pickle.loads(objs["worker"])
        state = objs["state"]
        filters = objs["filters"]
        self.filters = filters[self._policy_name]
        weights = state[self._policy_name]
        weights.pop("_optimizer_variables")
        self.policy.set_weights(weights)
        self.model = self.policy.model

        self.rnn_state = self.model.get_initial_state()
        self.rnn_state = [torch.reshape(self.rnn_state[0], shape=(1, -1))]

    def act(self, obs):

        # single infer
        obs = self._prep.transform(obs)
        obs = self.filters(obs, update=False)
        action, self.rnn_state, _ = self.policy.compute_actions([obs], self.rnn_state, explore=False)
        action = action[0]

        return action

class RLlibTorchFCPolicy(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, observation_space, action_space):
        self._checkpoint_path = load_path
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        flat_obs_space = self._prep.observation_space

        ray.init(ignore_reinit_error=True, local_mode=True)

        from utils.ppo_policy import PPOTorchPolicy as LoadPolicy
        from utils.fc_model import FCMultiNetwork
        ModelCatalog.register_custom_model("my_fc", FCMultiNetwork)
        config = ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG.copy()
        config['num_workers'] = 0
        config["model"]["custom_model"] = "my_fc"
        config['model']['free_log_std'] = True

        self.policy = LoadPolicy(flat_obs_space, self._action_space, config)
        objs = pickle.load(open(self._checkpoint_path, "rb"))
        objs = pickle.loads(objs["worker"])
        state = objs["state"]
        filters = objs["filters"]
        self.filters = filters[self._policy_name]
        weights = state[self._policy_name]
        weights.pop("_optimizer_variables")
        self.policy.set_weights(weights)
        self.model = self.policy.model

    def act(self, obs):

        # single infer
        obs = self._prep.transform(obs)
        obs = self.filters(obs, update=False)
        action, _, _ = self.policy.compute_actions([obs], explore=False)
        action = action[0]

        return action

class RLlibTorchMultiPolicy(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, observation_space, action_space):
        self._checkpoint_path = load_path
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        flat_obs_space = self._prep.observation_space

        ray.init(ignore_reinit_error=True, local_mode=True)

        from utils.ppo_policy import PPOTorchPolicy as LoadPolicy
        from utils.fc_model import FCMultiLayerNetwork
        ModelCatalog.register_custom_model("my_fc", FCMultiLayerNetwork)
        config = ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG.copy()
        config['num_workers'] = 0
        config["model"]["custom_model"] = "my_fc"
        config['model']['free_log_std'] = False

        self.policy = LoadPolicy(flat_obs_space, self._action_space, config)
        objs = pickle.load(open(self._checkpoint_path, "rb"))
        objs = pickle.loads(objs["worker"])
        state = objs["state"]
        filters = objs["filters"]
        self.filters = filters[self._policy_name]
        weights = state[self._policy_name]
        weights.pop("_optimizer_variables")
        self.policy.set_weights(weights)
        self.model = self.policy.model

    def act(self, obs):

        # single infer
        obs = self._prep.transform(obs)
        obs = self.filters(obs, update=False)
        action, _, _ = self.policy.compute_actions([obs], explore=False)
        action = action[0]

        return action