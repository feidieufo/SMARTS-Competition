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
        from utils.fc_model import FullyConnectedNetwork
        ModelCatalog.register_custom_model("my_fc", FullyConnectedNetwork)
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

class RLlibTorchFCDyskipPolicy(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, observation_space, action_space):
        self._checkpoint_path = load_path
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        flat_obs_space = self._prep.observation_space

        ray.init(ignore_reinit_error=True, local_mode=True)

        from utils.ppo_policy import PPOTorchPolicy as LoadPolicy
        from utils.fc_model import FullyConnectedNetwork
        ModelCatalog.register_custom_model("my_fc", FullyConnectedNetwork)
        ModelCatalog.register_custom_action_dist("my_dist", TorchDyDistribution)
        config = ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG.copy()
        config['num_workers'] = 0
        config["model"]["custom_model"] = "my_fc"
        config['model']['free_log_std'] = False
        config["model"]["custom_action_dist"] =  "my_dist"

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
        config["vf_share_layers"] = False
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

class RLlibTorchVFMultiPolicy(AgentPolicy):
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
        config["vf_share_layers"] = True
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


class RLlibTorchGRUMultiPolicy(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, observation_space, action_space):
        self._checkpoint_path = load_path
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        flat_obs_space = self._prep.observation_space

        ray.init(ignore_reinit_error=True, local_mode=True)

        from utils.ppo_policy import PPOTorchPolicy as LoadPolicy
        from utils.rnn_model import RNNMultiModel
        ModelCatalog.register_custom_model("my_rnn", RNNMultiModel)
        config = ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG.copy()
        config["vf_share_layers"] = True
        config['num_workers'] = 0
        config["model"]["custom_model"] = "my_rnn"
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

        self.rnn_state = self.model.get_initial_state()
        self.rnn_state = [torch.reshape(self.rnn_state[0], shape=(1, -1))]

    def act(self, obs):

        # single infer
        obs = self._prep.transform(obs)
        obs = self.filters(obs, update=False)
        action, self.rnn_state, _ = self.policy.compute_actions([obs], self.rnn_state, explore=False)
        action = action[0]

        return action