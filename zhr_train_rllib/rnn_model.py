from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

class RNNModel(RecurrentNetwork, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="rnn_model"):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.obs_size = _get_size(obs_space)
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        self.fc1 = nn.Linear(self.obs_size, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, num_outputs)

        self.value_branch = nn.Linear(self.rnn_hidden_dim, 1)
        self._cur_value = None        

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return [self.fc1.weight.new(1, self.rnn_hidden_dim).zero_().squeeze(0)]

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    @override(RecurrentNetwork)
    def forward(self, input_dict, state, seq_lens):
        output, new_state = self.forward_rnn(input_dict["obs_flat"].float(), state, seq_lens)
        return torch.reshape(output, [-1, self.num_outputs]), new_state        

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, hidden_state, seq_lens):
        x = nn.functional.relu(self.fc1(input_dict))
        h_in = hidden_state[0].reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        self._cur_value = self.value_branch(h).squeeze(1)
        return q, [h]


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size