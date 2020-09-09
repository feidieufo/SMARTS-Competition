from pathlib import Path

from utils.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

# from utils.saved_model import RLlibTFCheckpointPolicy, RLlibTFPolicy
from utils.saved_model_torch import RLlibTorchGRUPolicy, RLlibTorchGRUDVEPolicy

load_path = "checkpoint_376_dve144/checkpoint-376"
# load_path = "checkpoint_800_gru28/checkpoint-800"
# load_path = "checkpoint_792/checkpoint-792"
# load_path = "model"
# load_path = "checkpoint/checkpoint"

# agent_spec.policy_builder = lambda: RLlibTorchGRUPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

agent_spec.policy_builder = lambda: RLlibTorchGRUDVEPolicy(
    Path(__file__).parent / load_path,
    "PPO",
    "default_policy",
    OBSERVATION_SPACE,
    ACTION_SPACE,
)
