from pathlib import Path

from utils.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_multi import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

from utils.saved_model import RLlibTFCheckpointPolicy, RLlibTFPolicy, RLlibTFA2Policy, RLlibTFA2FilterPolicy
from utils.saved_model_torch import RLlibTorchGRUPolicy, RLlibTorchGRUDVEPolicy, RLlibTorchFCPolicy, RLlibTorchMultiPolicy, RLlibTorchGRUMultiPolicy, RLlibTorchVFMultiPolicy

# load_path = "checkpoint_260_itsa_ppo_baseline/checkpoint-260"
# load_path = "checkpoint_78_single_ppo_baseline_skip4_itsa/checkpoint-78"
# load_path = "checkpoint_680_single_ppo_baseline_merge/checkpoint-680"
# load_path = "checkpoint_348_fcmulti_a2_vf0.5/checkpoint-348"
# load_path = "checkpoint_78_single_ppo_baseline_merge/checkpoint-78"
# load_path = "checkpoint_328_single_ppo_baseline_all/checkpoint-328"
# load_path = "checkpoint_340_fcmulti_a2_skip2/checkpoint-340"
load_path = "checkpoint_214_baseline_sharp_mixroundaboutmerge/checkpoint-214"
# load_path = "checkpoint_348_fcmulti_a2_vf1_skip2/checkpoint-348"
# load_path = "checkpoint_598_fcmulti_a2_vf1/checkpoint-598"
# load_path = "checkpoint_392_tffc2_directline/checkpoint-392"
# load_path = "checkpoint_792/checkpoint-792"
# load_path = "checkpoint_912_a2/checkpoint-912"
# load_path = "checkpoint_912_gru_a2/checkpoint-912"
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

agent_spec.policy_builder = lambda: RLlibTFPolicy(
    Path(__file__).parent / load_path,
    "PPO",
    "default_policy",
    OBSERVATION_SPACE,
    ACTION_SPACE,
)
