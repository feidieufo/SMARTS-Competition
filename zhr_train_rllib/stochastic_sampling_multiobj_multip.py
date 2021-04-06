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

        x = inputs.mean(0)
        dist = torch.distributions.categorical.Categorical(logits=x[0])

        if explore:
            action = dist.sample()
            logp = action_dist.logp(action)

        else:
            action = dist.deterministic_sample()
            logp = torch.zeros(size=(inputs.shape[1], inputs.shape[0]), dtype=torch.float32)

        action = action.unsqueeze(0)
        return action, logp
