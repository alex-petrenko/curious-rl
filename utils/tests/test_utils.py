from unittest import TestCase

import numpy as np

from utils.decay import LinearDecay
from utils.doom.doom_utils import make_doom_env, env_by_name
from utils.utils import numpy_all_the_way, numpy_flatten


class TestDecay(TestCase):
    def test_linear_decay(self):
        with self.assertRaises(Exception):
            LinearDecay([])

        def chk(value, expected):
            self.assertAlmostEqual(value, expected)

        decay = LinearDecay([(0, 1)])
        chk(decay.at(0), 1)
        chk(decay.at(100), 1)

        decay = LinearDecay([(0, 0), (1000, 1)])
        chk(decay.at(-1), 0)
        chk(decay.at(0), 0)
        chk(decay.at(1000), 1)
        chk(decay.at(10000), 1)
        chk(decay.at(500), 0.5)
        chk(decay.at(450), 0.45)

        decay = LinearDecay([(0, 0), (1000, 1), (2000, 5)], staircase=0.1)
        chk(decay.at(-1), 0)
        chk(decay.at(0), 0)
        chk(decay.at(1000), 1)
        chk(decay.at(1500), 3)
        chk(decay.at(1501), 3)
        chk(decay.at(2000), 5)
        chk(decay.at(10000), 5)
        chk(decay.at(500), 0.5)
        chk(decay.at(401), 0.4)
        chk(decay.at(450), 0.4)
        chk(decay.at(499), 0.4)

        decay = LinearDecay([(0, 50), (100, 100)], staircase=100)
        chk(decay.at(0), 50)
        chk(decay.at(1), 50)
        chk(decay.at(99), 50)


class TestUtil(TestCase):
    def test_numpy_all_the_way(self):
        a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        lst = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])]
        numpy_arr = numpy_all_the_way(lst)
        self.assertTrue(np.array_equal(a, numpy_arr))

    def test_numpy_flatten(self):
        a = np.arange(9)
        lst = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])]
        flattened = numpy_flatten(lst)
        self.assertTrue(np.array_equal(a, flattened))


class TestDoom(TestCase):
    def test_doom_env(self):
        env = make_doom_env(env_by_name('maze'))
        self.assertIsNotNone(env)
