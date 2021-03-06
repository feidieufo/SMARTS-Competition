from gym.spaces import Discrete

import ray
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio, \
    PRIO_WEIGHTS, Q_SCOPE, Q_TARGET_SCOPE
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from .dqn_model_decompose import DQNTorchModel
from ray.rllib.agents.dqn.simple_q_torch_policy import TargetNetworkMixin
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration.parameter_noise import ParameterNoise
from ray.rllib.utils.torch_ops import huber_loss, reduce_mean_ignore_inf
from ray.rllib.utils import try_import_torch
import random
import numpy as np

torch, nn = try_import_torch()
F = None
if nn:
    F = nn.functional

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
class TorchMultiObjCategorical(TorchDistributionWrapper):
    """MultiCategorical distribution for MultiDiscrete action spaces."""

    @override(TorchDistributionWrapper)
    def __init__(self, inputs, model):
        # super().__init__(inputs, model)
        # If input_lens is np.ndarray or list, force-make it a tuple.
        self.inputs = inputs
        self.model = model
        self.cats = [
            torch.distributions.categorical.Categorical(logits=input_)
            for input_ in inputs
        ]

    @override(TorchDistributionWrapper)
    def sample(self):
        arr = [cat.sample() for cat in self.cats]
        self.last_sample = torch.stack(arr, dim=1)
        return self.last_sample

    @override(ActionDistribution)
    def deterministic_sample(self):
        arr = [torch.argmax(cat.probs, -1) for cat in self.cats]
        self.last_sample = torch.stack(arr, dim=1)
        return self.last_sample

    @override(TorchDistributionWrapper)
    def logp(self, actions):
        # # If tensor is provided, unstack it into list.
        if isinstance(actions, torch.Tensor):
            actions = torch.unbind(actions, dim=1)
        logps = torch.stack(
            [cat.log_prob(act) for cat, act in zip(self.cats, actions)])
        return torch.sum(logps, dim=0)

    @override(ActionDistribution)
    def multi_entropy(self):
        return torch.stack([cat.entropy() for cat in self.cats], dim=1)

    @override(TorchDistributionWrapper)
    def entropy(self):
        return torch.sum(self.multi_entropy(), dim=1)

    @override(ActionDistribution)
    def multi_kl(self, other):
        return torch.stack(
            [
                torch.distributions.kl.kl_divergence(cat, oth_cat)
                for cat, oth_cat in zip(self.cats, other.cats)
            ],
            dim=1,
        )

    @override(TorchDistributionWrapper)
    def kl(self, other):
        return torch.sum(self.multi_kl(other), dim=1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return np.sum(action_space.nvec)

class QLoss:
    def __init__(self,
                 q_t_selected,
                 q_tp1_best,
                 importance_weights,
                 rewards,
                 done_mask,
                 gamma=0.99,
                 n_step=1,
                 num_atoms=1,
                 v_min=-10.0,
                 v_max=10.0):

        if num_atoms > 1:
            raise ValueError("Torch version of DQN does not support "
                             "distributional Q yet!")

        q_tp1_best_masked = (1.0 - done_mask.unsqueeze(1)) * q_tp1_best
        
        # compute RHS of bellman equation
        q_t_selected_target = rewards + gamma**n_step * q_tp1_best_masked
        # q_t_selected_target = torch.clamp(q_t_selected_target, -1, 1)

        # compute the error (potentially clipped)
        self.td_error = q_t_selected - q_t_selected_target.detach()
        self.loss = torch.mean(
            importance_weights.float().unsqueeze(1) * huber_loss(self.td_error))
        self.stats = {
            "mean_q1": torch.mean(q_t_selected[:,0]),
            "min_q1": torch.min(q_t_selected[:,0]),
            "max_q1": torch.max(q_t_selected[:,0]),

            "mean_q2": torch.mean(q_t_selected[:,1]),
            "min_q2": torch.min(q_t_selected[:,1]),
            "max_q2": torch.max(q_t_selected[:,1]),

            "td_error": self.td_error,
            "mean_td_error": torch.mean(self.td_error),
        }


class ComputeTDErrorMixin:
    def __init__(self):
        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask,
                             importance_weights):
            input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_t})
            input_dict[SampleBatch.ACTIONS] = act_t
            input_dict[SampleBatch.REWARDS] = rew_t
            input_dict[SampleBatch.NEXT_OBS] = obs_tp1
            input_dict[SampleBatch.DONES] = done_mask
            input_dict[PRIO_WEIGHTS] = importance_weights

            # Do forward pass on loss to update td error attribute
            build_q_losses(self, self.model, None, input_dict)

            return self.q_loss.td_error

        self.compute_td_error = compute_td_error


def build_q_model_and_distribution(policy, obs_space, action_space, config):

    if not isinstance(action_space, Discrete):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for DQN.".format(action_space))

    if config["hiddens"]:
        # try to infer the last layer size, otherwise fall back to 256
        num_outputs = ([256] + config["model"]["fcnet_hiddens"])[-1]
        config["model"]["no_final_linear"] = True
    else:
        num_outputs = action_space.n

    # TODO(sven): Move option to add LayerNorm after each Dense
    #  generically into ModelCatalog.
    add_layer_norm = (
        isinstance(getattr(policy, "exploration", None), ParameterNoise)
        or config["exploration_config"]["type"] == "ParameterNoise")

    policy.q_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch",
        model_interface=DQNTorchModel,
        name=Q_SCOPE,
        dueling=config["dueling"],
        q_hiddens=config["hiddens"],
        use_noisy=config["noisy"],
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm,
        decompose_num=config["decompose_num"])

    policy.q_func_vars = policy.q_model.variables()

    policy.target_q_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch",
        model_interface=DQNTorchModel,
        name=Q_TARGET_SCOPE,
        dueling=config["dueling"],
        q_hiddens=config["hiddens"],
        use_noisy=config["noisy"],
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm,
        decompose_num=config["decompose_num"])

    policy.target_q_func_vars = policy.target_q_model.variables()

    return policy.q_model, TorchMultiObjCategorical


def get_distribution_inputs_and_class(policy,
                                      model,
                                      obs_batch,
                                      *,
                                      explore=True,
                                      is_training=False,
                                      **kwargs):
    q_vals = compute_q_values(policy, model, obs_batch, explore, is_training)
    # q_vals = sum(q_vals)
    # q_vals = q_vals[0] if isinstance(q_vals, tuple) else q_vals

    policy.q_values = q_vals
    return policy.q_values, TorchMultiObjCategorical, []  # state-out

def choose_tend_action(q_values):
    threshold = -0.1
    action_set_sub = [(q >= (torch.max(q, dim=-1)[0] + threshold).unsqueeze(1)).int() for q in q_values]
    valid = torch.ones_like(q_values[0]).int()

    for j in range(q_values[0].shape[0]):
        cur_valid = valid[j].clone().detach()
        last_valid = cur_valid.clone().detach()
        for (i,a_set) in enumerate(action_set_sub):
            cur_valid = cur_valid & a_set[j]

            if not cur_valid.bool().any():
                valid_q = q_values[i][j]*last_valid
                valid_q = torch.where(valid_q == 0.0, valid_q-10000.0, valid_q)
                a = torch.argmax(valid_q)
                cur_valid[a] = 1
                break
            last_valid = cur_valid
        valid[j] = cur_valid


    valid_q = valid * q_values[-1]
    valid_q = torch.where(valid_q == 0.0, valid_q-10000.0, valid_q)
    a = torch.argmax(valid_q, dim=-1)
    return a

    # a = torch.multinomial(valid.float(), num_samples=1)
    # return a.squeeze(dim=1)

def build_q_losses(policy, model, _, train_batch):
    config = policy.config
    # q network evaluation
    q_ts = compute_q_values(
        policy,
        policy.q_model,
        train_batch[SampleBatch.CUR_OBS],
        explore=False,
        is_training=True)

    # target q network evalution
    q_tp1s = compute_q_values(
        policy,
        policy.target_q_model,
        train_batch[SampleBatch.NEXT_OBS],
        explore=False,
        is_training=True)

    # q scores for actions which we know were selected in the given state.
    one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS],
                                  policy.action_space.n)
    # q_t_selected = torch.sum(q_t * one_hot_selection, 1)
    q_t_selected = [torch.sum(q_t * one_hot_selection, 1) for q_t in q_ts]

    # compute estimate of best possible value starting from state at t + 1
    if config["double_q"]:
        q_tp1_using_online_net = compute_q_values(
            policy,
            policy.q_model,
            train_batch[SampleBatch.NEXT_OBS],
            explore=False,
            is_training=True)
        # q_tp1_using_online_net = sum(q_tp1_using_online_net)
        # q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
        q_tp1_best_one_hot_selection = []
        for i in range(len(q_tp1_using_online_net)):
            q = q_tp1_using_online_net[0:i+1]
            q_tp1_best_using_online_net = choose_tend_action(q)
            a = F.one_hot(q_tp1_best_using_online_net,  policy.action_space.n)
            q_tp1_best_one_hot_selection.append(a)

        q_tp1_best = [torch.sum(q * a, 1) for (q,a) in zip(q_tp1s, q_tp1_best_one_hot_selection)]
    else:
        # qtp1 = sum(q_tp1s)
        q_tp1_best_one_hot_selection = []
        for i in range(len(q_tp1s)):
            q = q_tp1s[0:i+1]
            q_tp1_best_using_online_net = choose_tend_action(q)
            a = F.one_hot(q_tp1_best_using_online_net,  policy.action_space.n)
            q_tp1_best_one_hot_selection.append(a)
        # q_tp1_best = torch.sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        q_tp1_best = [torch.sum(q * a, 1) for (q,a) in zip(q_tp1s, q_tp1_best_one_hot_selection)]

    q_t_selected = torch.stack(q_t_selected, dim=1)
    q_tp1_best = torch.stack(q_tp1_best, dim=1)
    policy.q_loss = QLoss(q_t_selected, q_tp1_best, train_batch[PRIO_WEIGHTS],
                          train_batch[SampleBatch.REWARDS],
                          train_batch[SampleBatch.DONES].float(),
                          config["gamma"], config["n_step"],
                          config["num_atoms"], config["v_min"],
                          config["v_max"])

    return policy.q_loss.loss


def adam_optimizer(policy, config):
    return torch.optim.Adam(
        policy.q_func_vars, lr=policy.cur_lr, eps=config["adam_epsilon"])


def build_q_stats(policy, batch):
    return dict({
        "cur_lr": policy.cur_lr,
    }, **policy.q_loss.stats)


def setup_early_mixins(policy, obs_space, action_space, config):
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def after_init(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy, obs_space, action_space, config)
    # Move target net to device (this is done autoatically for the
    # policy.model, but not for any other models the policy has).
    policy.target_q_model = policy.target_q_model.to(policy.device)


def compute_q_values(policy, model, obs, explore, is_training=False):
    if policy.config["num_atoms"] > 1:
        raise ValueError("torch DQN does not support distributional DQN yet!")

    model_out, state = model({
        SampleBatch.CUR_OBS: obs,
        "is_training": is_training,
    }, [], None)

    advantages_or_q_values = model.get_advantages_or_q_values(model_out)

    if policy.config["dueling"]:
        state_value = model.get_state_value(model_out)
        advantages_mean = reduce_mean_ignore_inf(advantages_or_q_values, 1)
        advantages_centered = advantages_or_q_values - torch.unsqueeze(
            advantages_mean, 1)
        q_values = state_value + advantages_centered
    else:
        q_values = advantages_or_q_values

    return q_values


def grad_process_and_td_error_fn(policy, optimizer, loss):
    # Clip grads if configured.
    info = apply_grad_clipping(policy, optimizer, loss)
    # Add td-error to info dict.
    info["td_error"] = policy.q_loss.td_error
    return info


def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    return {"q_values": policy.q_values}


DQNTorchPolicy = build_torch_policy(
    name="DQNTorchPolicy",
    loss_fn=build_q_losses,
    get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    after_init=after_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ])