"""
Gym env wrappers that make the environment suitable for the RL algorithms.

"""


import cv2
import gym
import copy
import numpy as np

from gym import spaces, RewardWrapper
from collections import deque

from utils.utils import numpy_all_the_way


def unwrap_env(wrapped_env):
    return wrapped_env.unwrapped


def has_image_observations(env):
    """It's a heuristic."""
    return len(env.observation_space.shape) >= 2


class StackFramesWrapper(gym.core.Wrapper):
    """
    Gym env wrapper to stack multiple frames.
    Useful for training feed-forward agents on dynamic games.
    """

    def __init__(self, env, stack_past_frames):
        super(StackFramesWrapper, self).__init__(env)
        if len(env.observation_space.shape) not in [1, 2]:
            raise Exception('Stack frames works with vector observations and 2D single channel images')
        self._stack_past = stack_past_frames
        self._frames = None

        self._image_obs = has_image_observations(env)

        if self._image_obs:
            new_obs_space_shape = env.observation_space.shape + (stack_past_frames,)
        else:
            new_obs_space_shape = list(env.observation_space.shape)
            new_obs_space_shape[0] *= stack_past_frames

        self.observation_space = spaces.Box(
            0.0 if self._image_obs else env.observation_space.low[0],
            1.0 if self._image_obs else env.observation_space.high[0],
            shape=new_obs_space_shape,
            dtype=np.float32,
        )

    def _render_stacked_frames(self):
        if self._image_obs:
            return np.transpose(numpy_all_the_way(self._frames), axes=[1, 2, 0])
        else:
            return np.array(self._frames).flatten()

    def reset(self):
        observation = self.env.reset()
        self._frames = deque([np.zeros_like(observation)] * (self._stack_past - 1))
        self._frames.append(observation)
        return self._render_stacked_frames()

    def step(self, action):
        new_observation, reward, done, info = self.env.step(action)
        self._frames.popleft()
        self._frames.append(new_observation)
        return self._render_stacked_frames(), reward, done, info


class SkipAndStackFramesWrapper(StackFramesWrapper):
    """Wrapper for action repeat + stack multiple frames to capture dynamics."""

    def __init__(self, env, num_frames=4):
        super(SkipAndStackFramesWrapper, self).__init__(env, stack_past_frames=num_frames)
        self._skip_frames = num_frames

    def step(self, action):
        done = False
        total_reward, num_frames = 0, 0
        for i in range(self._skip_frames):
            new_observation, reward, done, info = self.env.step(action)
            num_frames += 1
            total_reward += reward
            self._frames.popleft()
            self._frames.append(new_observation)
            if done:
                break

        info['num_frames'] = num_frames
        return self._render_stacked_frames(), total_reward, done, info


class NormalizeWrapper(gym.core.Wrapper):
    """
    For environments with vector lowdim input.

    """

    def __init__(self, env):
        super(NormalizeWrapper, self).__init__(env)
        if len(env.observation_space.shape) != 1:
            raise Exception('NormalizeWrapper only works with lowdimensional envs')

        self.wrapped_env = env
        self._normalize_to = 1.0

        self._mean = (env.observation_space.high + env.observation_space.low) * 0.5
        self._max = env.observation_space.high

        self.observation_space = spaces.Box(
            -self._normalize_to, self._normalize_to, shape=env.observation_space.shape, dtype=np.float32,
        )

    def _normalize(self, obs):
        obs -= self._mean
        obs *= self._normalize_to / (self._max - self._mean)
        return obs

    def reset(self):
        observation = self.env.reset()
        return self._normalize(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._normalize(observation), reward, done, info

    @property
    def range(self):
        return [-self._normalize_to, self._normalize_to]


class ResizeAndGrayscaleWrapper(gym.core.Wrapper):
    """Resize observation frames to specified (w,h) and convert to grayscale."""

    def __init__(self, env, w, h):
        super(ResizeAndGrayscaleWrapper, self).__init__(env)
        self.observation_space = spaces.Box(0.0, 1.0, shape=[w, h], dtype=np.float32)
        self.w = w
        self.h = h

    def _observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.w, self.h), interpolation=cv2.INTER_AREA)
        obs = obs.astype(np.float32) / 255.0
        return obs

    def reset(self):
        return self._observation(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info


class RewardScalingWrapper(RewardWrapper):
    def __init__(self, env, scaling_factor):
        super(RewardScalingWrapper, self).__init__(env)
        self._scaling = scaling_factor
        self.reward_range = (r * scaling_factor for r in self.reward_range)

    def reward(self, reward):
        return reward * self._scaling

