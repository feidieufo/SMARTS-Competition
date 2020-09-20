from pathlib import Path

from utils.continuous_space_a2_multi import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

from utils.saved_model import RLlibTFCheckpointPolicy, RLlibTFPolicy, RLlibTFA2Policy, RLlibTFA2FilterPolicy
from utils.saved_model_torch import RLlibTorchGRUPolicy, RLlibTorchGRUDVEPolicy, RLlibTorchFCPolicy, RLlibTorchMultiPolicy

# load_path = "checkpoint_254_fcmultiline_directline/checkpoint-254"
# load_path = "checkpoint_376_dve144/checkpoint-376"
# load_path = "checkpoint_578_fcmulti_a2/checkpoint-578"
load_path = "checkpoint_348_fcmulti_a2_vf0.5/checkpoint-348"
# load_path = "checkpoint_392_tffc2_directline/checkpoint-392"
# load_path = "checkpoint_792/checkpoint-792"
# load_path = "checkpoint_912_a2/checkpoint-912"
# load_path = "checkpoint_912_gru_a2/checkpoint-912"
# load_path = "checkpoint_474_fc_a2/checkpoint-474"
# load_path = "model"
# load_path = "checkpoint/checkpoint"

# agent_spec.policy_builder = lambda: RLlibTorchGRUPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

# agent_spec.policy_builder = lambda: RLlibTorchFCPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

agent_spec.policy_builder = lambda: RLlibTorchMultiPolicy(
    Path(__file__).parent / load_path,
    "PPO",
    "default_policy",
    OBSERVATION_SPACE,
    ACTION_SPACE,
)

# agent_spec.policy_builder = lambda: RLlibTorchGRUDVEPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

# agent_spec.policy_builder = lambda: RLlibTFA2FilterPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )
