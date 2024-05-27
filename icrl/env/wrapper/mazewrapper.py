import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ConcatenateObservation(gym.Wrapper):
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.observation_space['observation'].shape[0]+2,), dtype="float32"
        )
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs = np.concatenate([obs["observation"], obs["desired_goal"]])
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.concatenate([obs["observation"], obs["desired_goal"]])
        return obs, info


class ConcatenateObservationNoGoal(gym.Wrapper):
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.observation_space['observation'].shape[0],), dtype="float32"
        )
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        return obs["observation"], rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["observation"], info
