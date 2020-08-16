from pathlib import Path

from utils.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

from utils.saved_model import RLlibTFCheckpointPolicy

load_path = "checkpoint_2/checkpoint-2"
# load_path = "model"
# load_path = "checkpoint/checkpoint"

agent_spec.policy_builder = lambda: RLlibTFCheckpointPolicy(
    Path(__file__).parent / load_path,
    "PPO",
    "default_policy",
    OBSERVATION_SPACE,
    ACTION_SPACE,
)
