from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import AppendBiasLayer

torch, nn = try_import_torch()

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

class RNNModel(RecurrentNetwork, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                model_config, name)
        nn.Module.__init__(self)

        self.obs_size = _get_size(obs_space)
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        self.free_log_std = model_config.get("free_log_std")
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2
        if self.free_log_std:
            self.log_std = torch.nn.Parameter(torch.as_tensor([0.0] * num_outputs))

        self.fc1 = nn.Linear(self.obs_size, self.rnn_hidden_dim)
        self.rnn = nn.GRU(self.rnn_hidden_dim, self.rnn_hidden_dim, batch_first=True)
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
        return torch.reshape(self._cur_value, [-1])   

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, hidden_state, seq_lens):
        x = nn.functional.relu(self.fc1(input_dict))
        x, h = self.rnn(x, torch.unsqueeze(hidden_state[0], 0))
        logits = self.fc2(x)
        if self.free_log_std:
            logstd = self.log_std.repeat(logits.shape[0], logits.shape[1], 1)
            logits = torch.cat([logits, logstd], dim=-1)

        self._cur_value = self.value_branch(x).squeeze(1)
        return logits, [torch.squeeze(h, 0)]        

def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size

class RNNDVEModel(RecurrentNetwork, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                model_config, name)
        nn.Module.__init__(self)

        self.obs_size = _get_size(obs_space)
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        self.free_log_std = model_config.get("free_log_std")
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2
        if self.free_log_std:
            self.log_std = torch.nn.Parameter(torch.as_tensor([0.0] * num_outputs))

        self.fc1 = nn.Linear(self.obs_size, self.rnn_hidden_dim)
        self.rnn = nn.GRU(self.rnn_hidden_dim, self.rnn_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, num_outputs)

        self.mu = nn.Linear(self.rnn_hidden_dim, 7)
        self.alpha = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, 7),
            nn.Softmax(dim=-1)
        )
        

        # self.value_branch = nn.Linear(self.rnn_hidden_dim, 1)
        self._cur_value = None        

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return [self.fc1.weight.new(1, self.rnn_hidden_dim).zero_().squeeze(0)]

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return torch.reshape(self._cur_value, [-1])   

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, hidden_state, seq_lens):
        x = nn.functional.relu(self.fc1(input_dict))
        x, h = self.rnn(x, torch.unsqueeze(hidden_state[0], 0))
        logits = self.fc2(x)
        if self.free_log_std:
            logstd = self.log_std.repeat(logits.shape[0], logits.shape[1], 1)
            logits = torch.cat([logits, logstd], dim=-1)

        self._cur_value = torch.sum(self.mu(x)*self.alpha(x), dim=-1)
        return logits, [torch.squeeze(h, 0)]   

class RNNMultiModel(RecurrentNetwork, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                model_config, name)
        nn.Module.__init__(self)

        self.obs_size = _get_size(obs_space)
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        self.free_log_std = model_config.get("free_log_std")
        num_outputs = num_outputs * 5
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2
        if self.free_log_std:
            self.log_std = torch.nn.Parameter(torch.as_tensor([0.0] * num_outputs))

        self.fc1 = nn.Linear(self.obs_size, self.rnn_hidden_dim)
        self.rnn = nn.GRU(self.rnn_hidden_dim, self.rnn_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, num_outputs)

        self.value_branch = nn.Linear(self.rnn_hidden_dim, 1)
        self._cur_value = None    

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        output, new_state = self.forward_rnn(
            add_time_dimension(
                input_dict["obs_flat"].float(), seq_lens, framework="torch"),
            state, seq_lens)
        return torch.reshape(output, [-1, self.num_outputs]), new_state    

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return [self.fc1.weight.new(1, self.rnn_hidden_dim).zero_().squeeze(0)]

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return torch.reshape(self._cur_value, [-1])   

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, hidden_state, seq_lens):
        x = nn.functional.relu(self.fc1(input_dict))
        x, h = self.rnn(x, torch.unsqueeze(hidden_state[0], 0))
        logits = self.fc2(x)
        if self.free_log_std:
            logstd = self.log_std.repeat(logits.shape[0], logits.shape[1], 1)
            logits = torch.cat([logits, logstd], dim=-1)

        self._cur_value = self.value_branch(x).squeeze(1)
        return logits, [torch.squeeze(h, 0)]  