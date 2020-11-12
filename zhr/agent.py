from pathlib import Path

# from utils.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_unroll_36 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_36 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_36_full_intersec_2 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_36_head_benchmark_dyskip import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
from utils.continuous_space_a3_36_head_benchmark import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_36_full import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_multi import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_multi_unroll import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.discrete_space_9 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.discrete_space_multi import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

from utils.saved_model import RLlibTFCheckpointPolicy, RLlibTFPolicy, RLlibTFA2Policy, RLlibTFA2FilterPolicy, RLlibLSTMPolicy
from utils.saved_model_torch import RLlibTorchGRUPolicy, RLlibTorchGRUDVEPolicy, RLlibTorchFCPolicy, RLlibTorchMultiPolicy, RLlibTorchGRUMultiPolicy, RLlibTorchVFMultiPolicy, RLlibTorchFCDyskipPolicy

# load_path = "model"
# load_path = "checkpoint/checkpoint"
load_path = "checkpoint_274/checkpoint-274"
load_path = "checkpoint_240_dyskip_three_penaty/checkpoint-240"
load_path = "checkpoint_321/checkpoint-321"

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

# agent_spec.policy_builder = lambda: RLlibTorchFCDyskipPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

# agent_spec.policy_builder = lambda: RLlibTorchMultiPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

# agent_spec.policy_builder = lambda: RLlibTorchVFMultiPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

# agent_spec.policy_builder = lambda: RLlibTorchGRUMultiPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )


# agent_spec.policy_builder = lambda: RLlibTorchGRUDVEPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

# agent_spec.policy_builder = lambda: RLlibTFA2Policy(
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

# agent_spec.policy_builder = lambda: RLlibTFPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

# agent_spec.policy_builder = lambda: RLlibLSTMPolicy(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )
