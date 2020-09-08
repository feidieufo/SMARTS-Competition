from pathlib import Path

from utils.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

# from utils.saved_model import RLlibTFCheckpointPolicy, RLlibTFPolicy
from utils.saved_model_torch import RLlibTorchGRUPolicy

load_path = "checkpoint_328_gru28/checkpoint-328"
# load_path = "checkpoint_792/checkpoint-792"
# load_path = "model"
# load_path = "checkpoint/checkpoint"

agent_spec.policy_builder = lambda: RLlibTorchGRUPolicy(
    Path(__file__).parent / load_path,
    "PPO",
    "default_policy",
    OBSERVATION_SPACE,
    ACTION_SPACE,
)
