"""
This file contains RLlib policy reload for evaluation usage, not for training.
"""
import os
import pickle

import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from smarts.core.agent import AgentPolicy

tf = try_import_tf()


#########################################
# restore checkpoint generated during training
# like load_Path = "checkpoint_200/checkpoint-200"
#########################################
class RLlibTFCheckpointPolicy(AgentPolicy):
    def __init__(
        self, load_path, algorithm, policy_name, observation_space, action_space
    ):
        self._checkpoint_path = load_path
        self._algorithm = algorithm
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._sess = None

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")

        if self._sess:
            return

        if self._algorithm == "PPO":
            from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy as LoadPolicy
        elif self._algorithm in ["A2C", "A3C"]:
            from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy as LoadPolicy
        elif self._algorithm == "PG":
            from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy as LoadPolicy
        elif self._algorithm == "DQN":
            from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy as LoadPolicy
        else:
            raise TypeError("Unsupport algorithm")

        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()

        with tf.name_scope(self._policy_name):
            # obs_space need to be flattened before passed to PPOTFPolicy
            flat_obs_space = self._prep.observation_space
            self.policy = LoadPolicy(flat_obs_space, self._action_space, {})
            objs = pickle.load(open(self._checkpoint_path, "rb"))
            objs = pickle.loads(objs["worker"])
            state = objs["state"]
            weights = state[self._policy_name]
            self.policy.set_weights(weights)

    def act(self, obs):
        if isinstance(obs, list):
            # batch infer
            obs = [self._prep.transform(o) for o in obs]
            action = self.policy.compute_actions(obs, explore=False)[0]
        else:
            # single infer
            obs = self._prep.transform(obs)
            action = self.policy.compute_actions([obs], explore=False)[0][0]

        return action


#########################################
# restore checkpoint exported at end
# like load_path = "checkpoint/checkpoint"
#########################################
class RLlibFinalCkptPolicy(AgentPolicy):
    def __init__(self, load_path, observation_space, action_space):
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._checkpoint_path = load_path

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")

        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()
        saver = tf.train.import_meta_graph(
            os.path.join(os.path.dirname(self._checkpoint_path), "model.meta")
        )
        saver.restore(
            self._sess, os.path.join(os.path.dirname(self._checkpoint_path), "model")
        )

        graph = tf.get_default_graph()

        if self.is_continuous:
            # These tensor names were found by inspecting the trained model
            # deterministic
            self.output_node = graph.get_tensor_by_name("default_policy/split:0")
            # add guassian noise
            # output_node = graph.get_tensor_by_name("default_policy/add:0")
        else:
            self.output_node = graph.get_tensor_by_name("default_policy/ArgMax:0")

        self.input_node = graph.get_tensor_by_name("default_policy/observation:0")

    def act(self, obs):
        if isinstance(obs, list):
            # batch infer
            obs = [self._prep.transform(o) for o in obs.values()]
            action = self._sess.run(self.output_node, feed_dict={self.input_node: obs})
        else:
            # single infer
            obs = self._prep.transform(obs)
            action = self._sess.run(
                self.output_node, feed_dict={self.input_node: [obs]}
            )[0]

        return action


#########################################
# restore model exported at end
# like load_path = "model"
#########################################
class RLlibTFModelPolicy(AgentPolicy):
    def __init__(self, load_path, observation_space, action_space):
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._path_to_model = load_path

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")

        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()
        tf.saved_model.load(self._sess, export_dir=self._path_to_model, tags=["serve"])

        graph = tf.get_default_graph()

        if self.is_continuous:
            # These tensor names were found by inspecting the trained model
            # deterministic
            self.output_node = graph.get_tensor_by_name("default_policy/split:0")
            # add guassian noise
            # output_node = graph.get_tensor_by_name("default_policy/add:0")
        else:
            self.output_node = graph.get_tensor_by_name("default_policy/ArgMax:0")

        self.input_node = graph.get_tensor_by_name("default_policy/observation:0")

    def act(self, obs):
        if isinstance(obs, list):
            # batch infer
            obs = [self._prep.transform(o) for o in obs]
            action = self._sess.run(self.output_node, feed_dict={self.input_node: obs})
        else:
            # single infer
            obs = self._prep.transform(obs)
            action = self._sess.run(
                self.output_node, feed_dict={self.input_node: [obs]}
            )[0]

        return action
