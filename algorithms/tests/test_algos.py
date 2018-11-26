import gym

from unittest import TestCase

from gym.wrappers import TimeLimit

from algorithms.agent import AgentLearner, AgentRandom
from algorithms.exploit import run_policy_loop
from algorithms.tests.test_wrappers import TEST_ENV_NAME
from utils.doom.doom_utils import make_doom_env, env_by_name


class TestAlgos(TestCase):
    def test_summary_step(self):
        params = AgentLearner.AgentParams('test')
        agent = AgentLearner(params)

        self.assertFalse(agent._should_write_summaries(0))
        self.assertTrue(agent._should_write_summaries(100 - 1))
        self.assertTrue(agent._should_write_summaries(200 - 1))

        self.assertTrue(agent._should_write_summaries(1002000 - 1))
        self.assertFalse(agent._should_write_summaries(1001000 - 1))
        self.assertFalse(agent._should_write_summaries(1000100 - 1))

    def test_run_loop(self):
        env = TimeLimit(make_doom_env(env_by_name(TEST_ENV_NAME)), max_episode_steps=50)

        agent = AgentRandom({}, env)
        run_policy_loop(agent, env, 1, 60)
