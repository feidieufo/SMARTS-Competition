from pathlib import Path

# from utils.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_unroll_36 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_36 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
from utils.continuous_space_a2_36_full_intersec_2 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_36_full import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_multi import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.continuous_space_a2_multi_unroll import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.discrete_space_9 import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
# from utils.discrete_space_multi import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

from utils.saved_model import RLlibTFCheckpointPolicy, RLlibTFPolicy, RLlibTFA2Policy, RLlibTFA2FilterPolicy, RLlibLSTMPolicy
from utils.saved_model_torch import RLlibTorchGRUPolicy, RLlibTorchGRUDVEPolicy, RLlibTorchFCPolicy, RLlibTorchMultiPolicy, RLlibTorchGRUMultiPolicy, RLlibTorchVFMultiPolicy

# load_path = "checkpoint_260_itsa_ppo_baseline/checkpoint-260"
# load_path = "checkpoint_78_single_ppo_baseline_skip4_itsa/checkpoint-78"
# load_path = "checkpoint_680_single_ppo_baseline_merge/checkpoint-680"
# load_path = "checkpoint_348_fcmulti_a2_vf0.5/checkpoint-348"
# load_path = "checkpoint_78_single_ppo_baseline_merge/checkpoint-78"
# load_path = "checkpoint_328_single_ppo_baseline_all/checkpoint-328"
# load_path = "checkpoint_340_fcmulti_a2_skip2/checkpoint-340"
# load_path = "checkpoint_214_baseline_sharp_mixroundaboutmerge/checkpoint-214"
# load_path = "checkpoint_240_baseline_sharp_mixroundaboutmerge/checkpoint-240"
# load_path = "checkpoint_430_grumulti_a2_skip2_seq60/checkpoint-430"
# load_path = "checkpoint_216_fcmulti_a2_vf1_soft/checkpoint-216"
# load_path = "checkpoint_430_fcmulti_a2_vf1_soft/checkpoint-430"
# load_path = "checkpoint_502_baseline_discrete9/checkpoint-502"
# load_path = "checkpoint_116_fcmulti_unroll_gamma0999_vf1_a2/checkpoint-116"
# load_path = "checkpoint_730_baseline_gamma0999/checkpoint-730"
# load_path = "checkpoint_556_fcmulti_a2_vf1_gamma0999_unroll/checkpoint-556"
# load_path = "checkpoint_300_fcmulti_a2_vf1_gamma0995_unroll_lr/checkpoint-300"
# load_path = "checkpoint_176_baseline_gamma_36_meanstd_complete_gradclip/checkpoint-176"
# load_path = "checkpoint_376_baseline_gamma_36_meanstd_complete_gradclip/checkpoint-376"
# load_path = "checkpoint_268_fc2_gamma0995_dual_36/checkpoint-268"
# load_path = "checkpoint_328_fc2_gamma0995_dual_36/checkpoint-328"
# load_path = "checkpoint_222_fc_gamma0995_36_center/checkpoint-222"
# load_path = "checkpoint_560_fc2_dual_clip_gamma0999_36/checkpoint-560"
# load_path = "checkpoint_651_fc2_dual_clip_gamma0999_36/checkpoint-651"
# load_path = "checkpoint_232_fc2_dual_clip_gamma0997_36/checkpoint-232"
# load_path = "checkpoint_288_fc2_dual_clip_gamma0997_36/checkpoint-288"
# load_path = "checkpoint_494_fc2_dual_clip_gamma0997_36_l2/checkpoint-494"
# load_path = "checkpoint_656_baseline_fc2_gamma0995_36/checkpoint-656"
# load_path = "checkpoint_1122_fc2_dual_clip_gamma0999_36/checkpoint-1122"
# load_path = "checkpoint_272_fc2_gamma0997_dual_36_fullinter/checkpoint-272"
# load_path = "checkpoint_332_fc2_gamma0997_dual_36_full/checkpoint-332"
# load_path = "checkpoint_1334_fc2_dual_clip_gamma0999_36/checkpoint-1334"
# load_path = "checkpoint_350_fc2_dual_clip_gamma0997_36_more/checkpoint-350"
# load_path = "checkpoint_426_baseline_fc2_dual_gamma0997_full/checkpoint-426"
# load_path = "checkpoint_1424_fc2_dual_gamma0997_36_cbest_more/checkpoint-1424"
# load_path = "checkpoint_1610_fc2_dual_gamma0997_36/checkpoint-1610"
# load_path = "checkpoint_352_fc2_dual_gamma0998_36_more_full/checkpoint-352"
# load_path = "checkpoint_1022_fc2_dual_clip_gamma0999_36/checkpoint-1022"
# load_path = "checkpoint_1198_fc2_dual_clip_gamma0999_36/checkpoint-1198"
# load_path = "checkpoint_212_fc2_gamma998_dual_36_stick/checkpoint-212"
# load_path = "checkpoint_1854_fc2_dual_gamma0999_36/checkpoint-1854"
# load_path = "checkpoint_1924_fc2_dual_gamma0999_36/checkpoint-1924"
# load_path = "checkpoint_1792_fc2_dual_gamma0997_36_cbest_more/checkpoint-1792"
# load_path = "checkpoint_180_baseline_dual_gamma1_fullinter/checkpoint-180"
# load_path = "checkpoint_342_fc2_dual_gamma0998_36_stick/checkpoint-342"
# load_path = "checkpoint_1450_fc2_dual_gamma0998_36_fullinter_stick/checkpoint-1450"
# load_path = "checkpoint_1824_fc2_dual_gamma0998_36_fullinter_stick/checkpoint-1824"
# load_path = "checkpoint_388_fc2_dual_gamma0999_36_full_intersec_more/checkpoint-388"
# load_path = "checkpoint_524_fc2_dual_gamma0999_full_intersec_36_more/checkpoint-524"
# load_path = "checkpoint_2526_fc2_dual_gamma0999_36/checkpoint-2526"
# load_path = "checkpoint_264_fc2_gamma0999_dual_36_stick/checkpoint-264"
# load_path = "checkpoint_1132_fc2_dual_gamma0999_fullinter_r_more_36/checkpoint-1132"
# load_path = "checkpoint_176_fc2_dual_gamma0999_fullinter_intersec_l2_36/checkpoint-176"
# load_path = "checkpoint_288_fc2_dual_gamma0999_fullinter_intersec_36/checkpoint-288"
# load_path = "checkpoint_498_fc2_dual_gamma0999_fullinter_36_stick/checkpoint-498"
# load_path = "checkpoint_364_fc2_dual_gamma0999_fullinter_intersec_l2_36/checkpoint-364"
# load_path = "checkpoint_718_fc2_dual_gamma0999_fullinter_intersec_36/checkpoint-718"
# load_path = "checkpoint_324_fc2_dual_gamma1_fullinter_intersec_36/checkpoint-324"
# load_path = "checkpoint_502_fc2_dual_gamma1_fullinter_intersec_36/checkpoint-502"
# load_path = "checkpoint_624_fc2_dual_gamma1_fullinter_intersec_36/checkpoint-624"
# load_path = "checkpoint_468/checkpoint-468"
# load_path = "checkpoint_512/checkpoint-512"
# load_path = "checkpoint_366_fc2_dual_gamma0999_fullinter_intersec3_36/checkpoint-366"
# load_path = "checkpoint_728_fc2_dual_gamma1_fullinter_intersec_36/checkpoint-728"
# load_path = "checkpoint_780_fc2_dual_gamma1_fullinter_intersec_36/checkpoint-780"
load_path = "checkpoint_322_fc2_gamma0999_dual_fullinter_intersec3_36_stick/checkpoint-322"
# load_path = "checkpoint_742_fc2_dual_gamma0999_fullinter_intersec_36_stick/checkpoint-742"
# load_path = "checkpoint_842_fc2_dual_gamma0999_fullinter_intersec3_36/checkpoint-842"
# load_path = "checkpoint_524_fc2_dual_gamma1_fullinter_intersec3_36/checkpoint-524"
# load_path = "checkpoint_942_fc2_dual_gamma1_fullinter_intersec_36/checkpoint-942"
# load_path = "checkpoint_1010_fc2_dual_gamma1_fullinter_intersec_36/checkpoint-1010"
# load_path = "checkpoint_736_fc2_dual_gamma1_fullinter_intersec3_36/checkpoint-736"
# load_path = "checkpoint_456_fc2_dual_gamma0999_fullinter_intersec3_36_lr5e-5/checkpoint-456"
# load_path = "checkpoint_794_fc2_dual_gamma0999_fullinter_intersec_36_stick/checkpoint-794"
# load_path = "checkpoint_832_fc2_dual_gamma1_fullinter_intersec_36/checkpoint-832"
# load_path = "checkpoint_874_fc2_dual_gamma1_fullinter_intersec_36/checkpoint-874"
# load_path = "checkpoint_512_fc2_dual_gamma0999_fullinter_intersec_36_stick/checkpoint-512"
# load_path = "checkpoint_1106_fc2_dual_gamma0999_fullinter_intersec_36/checkpoint-1106"
# load_path = "checkpoint_1250_fc2_dual_gamma1_fullinter_intersec_36/checkpoint-1250"
# load_path = "checkpoint_612_fc2_dual_gamma0999_fullinter_intersec3_r_36/checkpoint-612"
# load_path = "checkpoint_850_fc2_dual_gamma0999_fullinter_intersec3_36/checkpoint-850"
# load_path = "checkpoint_312_fc2_gamma0999_68dual_fullinter_intersec_36_stick/checkpoint-312"
# load_path = "checkpoint_1544_fc2_dual_gamma0999_fullinter_r_more_36/checkpoint-1544"
# load_path = "checkpoint_824_fc2_dual_gamma0999_fullinter_intersec_l2_36/checkpoint-824"
# load_path = "checkpoint_1808_fc2_dual_gamma0999_fullinter_meanstd_36/checkpoint-1808"
# load_path = "checkpoint_670_baseline_fc2_gamma0995_unroll36/checkpoint-670"
# load_path = "checkpoint_274_baseline_fc2_gamma0995_unroll_36/checkpoint-274"
# load_path = "checkpoint_1230_fcmulti_a2_gamma0999/checkpoint-1230"
# load_path = "checkpoint_504_fcmulti_a2_gamma0999/checkpoint-504"
# load_path = "checkpoint_382_baseline_gamma0995_36/checkpoint-382"
# load_path = "checkpoint_366_fc2_gamma0995_stickaction/checkpoint-366"
# load_path = "checkpoint_290_fc2_gamma0995_l21e-4_36/checkpoint-290"
# load_path = "checkpoint_718_fcmulti_gamma0999_a2/checkpoint-718"
# load_path = "checkpoint_182_fcmulti_gamma0999_a2/checkpoint-182"
# load_path = "checkpoint_302_baseline_lstm_gamma0999_unroll_soft/checkpoint-302"
# load_path = "checkpoint_158_fc_a2_skip3/checkpoint-158"
# load_path = "checkpoint_780_fcmulti_discrete/checkpoint-780"
# load_path = "checkpoint_1126_grumulti_a2_skip2/checkpoint-1126"
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

agent_spec.policy_builder = lambda: RLlibTorchFCPolicy(
    Path(__file__).parent / load_path,
    "PPO",
    "default_policy",
    OBSERVATION_SPACE,
    ACTION_SPACE,
)

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
