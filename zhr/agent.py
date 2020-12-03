from pathlib import Path

# from utils.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_unroll_36 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_36 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_36_full_intersec_2 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_36_head_benchmark_dyskip import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
from utils.continuous_space_a3_36_head_benchmark import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.discrete_space_36_head_benchmark_neighbor import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a3_36_head_benchmark_repeat import agent_spec, OBSERVATION_SPACE, ACTION_SPACE


from utils.saved_model import RLlibTFCheckpointPolicy, RLlibTFPolicy, RLlibTFA2Policy, RLlibTFA2FilterPolicy, RLlibLSTMPolicy 
from utils.saved_model_torch import RLlibTorchGRUPolicy, RLlibTorchGRUDVEPolicy, RLlibTorchFCPolicy, \
RLlibTorchMultiPolicy, RLlibTorchGRUMultiPolicy, RLlibTorchVFMultiPolicy, \
RLlibTorchFCDyskipPolicy, RLlibTorchLstmDistPolicy, RLlibTorchLstmPolicy, RLlibTorchFCDecomposePolicy

# load_path = "model"
# load_path = "checkpoint/checkpoint"
load_path = "checkpoint_274/checkpoint-274"
load_path = "checkpoint_240_dyskip_three_penaty/checkpoint-240"
load_path = "checkpoint_200_its_two_a3/checkpoint-200"
# load_path = "checkpoint_321_its_two_a3_decompose/checkpoint-321"
load_path = "checkpoint_321_its_two_a3_crashflag_del200/checkpoint-321"
load_path = "checkpoint_321_its_simple_a3_ogm/checkpoint-321"
# load_path = "checkpoint_200_its_simple_discrete_neighbor/checkpoint-200"
# load_path = "checkpoint_380_its_repeat_two_lstm_a3/checkpoint-380"
# load_path = "checkpoint_140_its_lstm_two_a3/checkpoint-140"

# agent_spec.policy_builder = lambda: RLlibTorchGRUPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

agent_spec.policy_builder = lambda: RLlibTorchFCPolicy(
    Path(__file__).parent / load_path,
    "PPO",
    "default_policy",
    OBSERVATION_SPACE,
    ACTION_SPACE,
)

# agent_spec.policy_builder = lambda: RLlibTorchFCDecomposePolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

# agent_spec.policy_builder = lambda: RLlibTorchFCDyskipPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

# agent_spec.policy_builder = lambda: RLlibTorchLstmDistPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

# agent_spec.policy_builder = lambda: RLlibTorchLstmPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )