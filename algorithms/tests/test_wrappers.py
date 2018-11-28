import gym
import numpy as np

from unittest import TestCase

from algorithms.agent import AgentRandom
from algorithms.env_wrappers import NormalizeWrapper, StackFramesWrapper, unwrap_env, ResizeAndGrayscaleWrapper, \
    SkipAndStackFramesWrapper, TimeLimitWrapper
from algorithms.multi_env import MultiEnv
from utils.doom.doom_utils import make_doom_env, DOOM_W, DOOM_H, env_by_name

TEST_ENV_NAME = 'maze'
TEST_ENV = env_by_name(TEST_ENV_NAME).env_id
TEST_LOWDIM_ENV = 'CartPole-v0'


class TestWrappers(TestCase):
    def test_normalize(self):
        env = gym.make(TEST_LOWDIM_ENV)
        self.assertEqual(len(env.observation_space.shape), 1)

        def check_range(test, o):
            for i in range(len(o)):
                test.assertLessEqual(o[i], env.observation_space.high[i])
                test.assertGreaterEqual(o[i], env.observation_space.low[i])

        obs = env.reset()
        check_range(self, obs)

        agent = AgentRandom({}, env)
        obs, _, _, _ = env.step(agent.best_action(obs))
        check_range(self, obs)

        env = NormalizeWrapper(env)
        obs = env.reset()
        check_range(self, obs)
        agent = AgentRandom({}, env)
        obs, _, _, _ = env.step(agent.best_action(obs))
        check_range(self, obs)

    def test_stacked_lowdim(self):
        orig_env = gym.make(TEST_LOWDIM_ENV)
        ndim = orig_env.observation_space.shape[0]

        stack = 5
        env = StackFramesWrapper(orig_env, stack)
        self.assertEqual(len(env.observation_space.shape), len(orig_env.observation_space.shape))
        self.assertEqual(env.observation_space.shape[0], orig_env.observation_space.shape[0] * stack)
        obs = env.reset()
        self.assertEqual(ndim * stack, len(obs))

    def test_stacked_pixels(self):
        orig_env = gym.make(TEST_ENV)
        env = ResizeAndGrayscaleWrapper(orig_env, DOOM_W, DOOM_H)

        stack = 5
        env = StackFramesWrapper(env, stack)
        env.reset()

    def test_repeat(self):
        env = gym.make(TEST_ENV)
        env = ResizeAndGrayscaleWrapper(env, DOOM_W, DOOM_H)
        env = SkipAndStackFramesWrapper(env, num_frames=4)
        env.reset()
        _, _, _, info = env.step(0)
        self.assertEqual(info['num_frames'], 4)

    def test_timer(self):
        env = gym.make(TEST_ENV)
        timelimit = 50
        env = TimeLimitWrapper(env, timelimit, 0)
        env.reset()

        i = done = info = None
        for i in range(timelimit + 1):
            _, _, done, info = env.step(0)
        self.assertEqual(i, timelimit)
        self.assertTrue(done)
        self.assertIn(TimeLimitWrapper.terminated_by_timer, info)
        self.assertTrue(info[TimeLimitWrapper.terminated_by_timer])

        env.close()
        env = gym.make(TEST_ENV)
        env = TimeLimitWrapper(env, timelimit, 3)
        done = False
        steps = 0
        while not done:
            _, _, done, info = env.step(0)
            steps += 1
        self.assertGreaterEqual(steps, timelimit - 3)
        self.assertLessEqual(steps, timelimit + 3)
        self.assertTrue(info[TimeLimitWrapper.terminated_by_timer])

    def test_unwrap(self):
        env = make_doom_env(env_by_name(TEST_ENV_NAME))
        unwrapped = unwrap_env(env)
        self.assertIsNot(type(unwrapped), gym.core.Wrapper)


class TestMultiEnv(TestCase):
    stacked_frames = 3

    @staticmethod
    def make_env_func():
        env = make_doom_env(env_by_name(TEST_ENV_NAME))
        return env

    def test_multi_env(self):
        """Just a basic sanity check test."""
        num_envs = 8
        multi_env = MultiEnv(
            num_envs=num_envs,
            num_workers=4,
            make_env_func=self.make_env_func,
            stats_episodes=10,
        )
        obs = multi_env.initial_obs()

        self.assertEqual(len(obs), num_envs)

        num_different = 0
        for i in range(1, len(obs)):
            if not np.array_equal(obs[i - 1], obs[i]):
                num_different += 1

        # By pure chance some of the environments might be identical even with different seeds, but definitely not
        # all of them!
        self.assertGreater(num_different, len(obs) // 2)

        for i in range(20):
            obs, rewards, dones, infos = multi_env.step([0] * num_envs)
            self.assertEqual(len(obs), num_envs)
            self.assertEqual(len(rewards), num_envs)
            self.assertEqual(len(dones), num_envs)
            self.assertEqual(len(infos), num_envs)

        multi_env.close()
