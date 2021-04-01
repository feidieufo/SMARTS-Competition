from pathlib import Path

from zhr_train_rllib.utils.discrete_space_36_head_benchmark_ogm_9_decompose2_multiobj import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

from zhr.utils.saved_model_simple import RLlibTorchFCMultiobj, RLlibTorchFCMultiV

load_path = "checkpoint_274/checkpoint-274"
load_path = "checkpoint_240_dyskip_three_penaty/checkpoint-240"
load_path = "checkpoint_200_its_two_a3/checkpoint-200"
# load_path = "checkpoint_321_its_two_a3_decompose/checkpoint-321"
load_path = "checkpoint_321_its_two_a3_crashflag_del200/checkpoint-321"
load_path = "checkpoint_321_its_simple_a3_ogm/checkpoint-321"
load_path = "/root/workspace/results/train_multi_scenario_fc3_its_simple_discrete_ppo_decompose2_multiobj/later_multi_scenarios_head_nopenalty_simple_ogm_discrete9_ppo_decompose2_multiobj_tau05_epsilon2-PPO-1/PPO_TORCH_MultiEnv_3_2021-03-21_10-56-036p92kv82/checkpoint_161/checkpoint-161"
load_path = "/root/workspace/results/train_multi_scenario_fc3_its_simple_discrete_ppo_decompose3/later_multi_scenarios_head_nopenalty_simple_ogm_discrete9_ppo_decompose2-PPO-1/PPO_TORCH_MultiEnv_0_2021-03-25_18-15-308r2f4nay/checkpoint_120/checkpoint-120"

# agent_spec.policy_builder = lambda: RLlibTorchFCMultiobj(
#     Path(__file__).parent / load_path,
#     "PPO",
#     "default_policy",
#     OBSERVATION_SPACE,
#     ACTION_SPACE,
# )

agent_spec.policy_builder = lambda: RLlibTorchFCMultiV(
    Path(__file__).parent / load_path,
    "PPO",
    "default_policy",
    OBSERVATION_SPACE,
    ACTION_SPACE,
)
