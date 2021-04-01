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
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.models.action_dist import ActionDistribution
from typing import Union
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType, get_variable
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.schedules import Schedule, PiecewiseSchedule

class StochasticSampling(Exploration):
    """An exploration that simply samples from a distribution.

    The sampling can be made deterministic by passing explore=False into
    the call to `get_exploration_action`.
    Also allows for scheduled parameters for the distributions, such as
    lowering stddev, temperature, etc.. over time.
    """

    def __init__(self, action_space, *, framework: str, model: ModelV2,
                 initial_epsilon=1.0,
                 final_epsilon=0.05,
                 epsilon_timesteps=int(1e5),
                 epsilon_schedule=None,
                 **kwargs):
        """Initializes a StochasticSampling Exploration object.

        Args:
            action_space (Space): The gym action space used by the environment.
            framework (str): One of None, "tf", "torch".
        """
        assert framework is not None
        super().__init__(
            action_space, model=model, framework=framework, **kwargs)

        self.epsilon_schedule = \
            from_config(Schedule, epsilon_schedule, framework=framework) or \
            PiecewiseSchedule(
                endpoints=[
                    (0, initial_epsilon), (epsilon_timesteps, final_epsilon)],
                outside_value=final_epsilon,
                framework=self.framework)

        self.last_timestep = get_variable(
            0, framework=framework, tf_name="timestep")

    @override(Exploration)
    def get_exploration_action(self,
                               *,
                               action_distribution: ActionDistribution,
                               timestep: Union[int, TensorType],
                               explore: bool = True):
        if self.framework == "torch":
            return self._get_torch_exploration_action(action_distribution,
                                                      explore, timestep)
        else:
            raise ValueError("TF version does not support "
                    "multiobj stochastic_sampling yet!")


    def _get_torch_exploration_action(self, action_dist, explore, timestep):
        self.last_timestep = timestep
        inputs = action_dist.inputs
        obj_num = inputs.shape[0]
        assert inputs.shape[1]==1
        action_p = torch.softmax(inputs, dim=-1)
        epsilon = self.epsilon_schedule(self.last_timestep)
        threshold = -0.5
        a = torch.max(action_p[0], dim=-1)

        y = (action_p[0] >=  torch.max(action_p[0], dim=-1)[0])
        action_p_sup = [(action_p[i] >=  torch.max(action_p[i], dim=-1)[0] + threshold).int() for i in range(obj_num)]
        valid = torch.ones_like(action_p_sup[0]).int()
        for (i,action) in enumerate(action_p_sup):
            last = valid
            valid = valid & action

            if not valid.bool().any():
                valid = last
                break
        inputs_logits = inputs[i]
        x = torch.where(valid!=0, inputs_logits, torch.tensor(0.0, dtype=torch.float32))
        pos = x!=0.0
        xx = x[pos]
        dist = torch.distributions.categorical.Categorical(logits=xx)
        label = torch.arange(0, 9).int().reshape(pos.shape)
        label = label[pos]

        if explore:
            action = dist.sample()
            action = label[action]
            logp = action_dist.logp(action)
            # epsilon = self.epsilon_schedule(self.last_timestep)
            # c = random.uniform(0, 1)
            # pos = random.randint(0, 1)
            # d = torch.distributions.categorical.Categorical(logits=inputs[pos][0])
            # if c < epsilon:
            #     action = d.sample()
            #     logp = action_dist.logp(action)
            #     logp1 = math.log(epsilon*1.0/inputs.shape[0])
            #     logp = logp + logp1
            # else:
            #     action = dist.sample()
            #     action = label[action]
            #     logp = action_dist.logp(action)
            #     logp1 = math.log(1-epsilon)
            #     logp = logp + logp1
        else:
            action = xx.argmax(dim=-1)
            action = label[action]
            logp = torch.zeros(size=(inputs.shape[1], inputs.shape[0]), dtype=torch.float32)

        action = action.unsqueeze(0)
        return action, logp

class RLlibTorchFCMultiobj(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, observation_space, action_space):
        self._checkpoint_path = load_path
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        flat_obs_space = self._prep.observation_space

        ray.init(ignore_reinit_error=True, local_mode=True)

        from zhr_train_rllib.ppo_policy_modeldist_multiv_multiobj import PPOTorchPolicy as LoadPolicy
        config = ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG.copy()
        config['num_workers'] = 0
        config['model']['free_log_std'] = False
        config["exploration_config"]["type"] = "zhr.utils.saved_model_simple.StochasticSampling"

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


class RLlibTorchFCMultiV(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, observation_space, action_space):
        self._checkpoint_path = load_path
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        flat_obs_space = self._prep.observation_space

        ray.init(ignore_reinit_error=True, local_mode=True)

        from zhr_train_rllib.ppo_policy_modeldist_multiv import PPOTorchPolicy as LoadPolicy
        config = ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG.copy()
        config['num_workers'] = 0
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