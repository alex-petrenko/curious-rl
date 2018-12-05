import numpy as np

from unittest import TestCase

from algorithms.agent import AgentLearner, AgentRandom
from algorithms.algo_utils import RunningMeanStd, extract_keys
from algorithms.env_wrappers import TimeLimitWrapper
from algorithms.exploit import run_policy_loop
from algorithms.tests.test_wrappers import TEST_ENV_NAME
from utils.doom.doom_utils import make_doom_env, env_by_name
from utils.utils import log


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
        env = TimeLimitWrapper(make_doom_env(env_by_name(TEST_ENV_NAME), mode='test'), 50, 0)
        agent = AgentRandom({}, env)
        run_policy_loop(agent, env, 1, 200)


class TestAlgoUtils(TestCase):
    def test_running_mean_std(self):
        running_mean_std = RunningMeanStd(max_past_samples=100000)

        true_mu, true_sigma, batch_size = -1, 3, 256

        x = np.random.normal(true_mu, true_sigma, batch_size)

        running_mean_std.update(x)

        # after 1 batch we should have almost the exact same
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        self.assertAlmostEqual(running_mean_std.mean, batch_mean, places=5)
        self.assertAlmostEqual(running_mean_std.var, batch_var, places=5)
        self.assertAlmostEqual(running_mean_std.count, batch_size, places=3)

        # after many batches we should have an accurate estimate
        for _ in range(1000):
            x = np.random.normal(true_mu, true_sigma, batch_size)
            running_mean_std.update(x)

        log.info('estimated mean %.2f variance %.2f', running_mean_std.mean, running_mean_std.var)
        self.assertAlmostEqual(running_mean_std.mean, true_mu, places=0)
        self.assertAlmostEqual(running_mean_std.var, true_sigma ** 2, places=0)

    def test_extract_keys(self):
        test_obs = [{'obs1': 1, 'obs2': 2}, {'obs1': 3, 'obs2': 4}]
        obs1, obs2 = extract_keys(test_obs, 'obs1', 'obs2')
        self.assertEqual(obs1, [1, 3])
        self.assertEqual(obs2, [2, 4])
