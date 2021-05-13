from typing import Union

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils import try_import_tree
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.schedules import Schedule, PiecewiseSchedule
import random
import math
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType, get_variable

tf = try_import_tf()
torch, _ = try_import_torch()
tree = try_import_tree()


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
            raise ValueError("Torch version does not support "
                    "multiobj stochastic_sampling yet!")


    def _get_torch_exploration_action(self, action_dist, explore, timestep):
        self.last_timestep = timestep
        inputs = action_dist.inputs
        obj_num = inputs.shape[0]
        assert inputs.shape[1]==1
        action_p = torch.softmax(inputs, dim=-1)
        epsilon = self.epsilon_schedule(self.last_timestep)
        threshold = [-epsilon, -1]

        explorei = obj_num
        if random.random() < epsilon:
            explorei = random.randrange(obj_num)

        y = (action_p[0] >=  torch.max(action_p[0], dim=-1)[0])
        action_p_sup = [(action_p[i] >=  torch.max(action_p[i], dim=-1)[0] + threshold[i]).int() for i in range(obj_num)]
        valid = torch.ones_like(action_p_sup[0]).int()
        for (i,action) in enumerate(action_p_sup):
            last = valid
            valid = valid & action

            if not valid.bool().any():
                valid = last
                break

            if explorei == i:
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
            logp = dist.log_prob(action)
            logp = logp * torch.ones(size=(inputs.shape[1], inputs.shape[0]), dtype=torch.float32)
            action = label[action]

        else:
            action = dist.deterministic_sample()
            action = label[action]
            logp = torch.zeros(size=(inputs.shape[1], inputs.shape[0]), dtype=torch.float32)

        action = action.unsqueeze(0)
        return action, logp
