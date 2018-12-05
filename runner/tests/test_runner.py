import logging

import numpy as np

from unittest import TestCase

from runner.run_descriptions.run_description import ParamGrid, ParamList, Experiment, RunDescription
from runner.run_many import run_many


class TestParams(TestCase):
    def test_param_list(self):
        params = [
            {'p1': 1, 'p2': 'a'},
            {'p2': 'b', 'p4': 'test'},
        ]
        param_list = ParamList(params)
        param_combinations = list(param_list.generate_params(randomize=False))

        for i, combination in enumerate(params):
            self.assertEqual(combination, param_combinations[i])

    def test_param_grid(self):
        grid = ParamGrid([
            ('p1', [0, 1]),
            ('p2', ['a', 'b', 'c']),
            ('p3', [None, {}]),
        ])

        param_combinations = grid.generate_params(randomize=True)
        for p in param_combinations:
            for key in ('p1', 'p2', 'p3'):
                self.assertIn(key, p)

        param_combinations = list(grid.generate_params(randomize=False))
        self.assertEqual(param_combinations[0], {'p1': 0, 'p2': 'a', 'p3': None})
        self.assertEqual(param_combinations[1], {'p1': 0, 'p2': 'a', 'p3': {}})
        self.assertEqual(param_combinations[-2], {'p1': 1, 'p2': 'c', 'p3': None})
        self.assertEqual(param_combinations[-1], {'p1': 1, 'p2': 'c', 'p3': {}})


class TestRunner(TestCase):
    def test_experiment(self):
        params = ParamGrid([('p1', [3.14, 2.71]), ('p2', ['a', 'b', 'c'])])
        cmd = 'python super_rl.py'
        ex = Experiment('test', cmd, params.generate_params(randomize=False))
        cmds = ex.generate_experiments()
        for command, name in cmds:
            self.assertTrue(command.startswith(cmd))
            self.assertTrue(name.startswith('test'))

    def test_descr(self):
        params = ParamGrid([('p1', [3.14, 2.71]), ('p2', ['a', 'b', 'c'])])
        experiments = [
            Experiment('test1', 'python super_rl1.py', params.generate_params(randomize=False)),
            Experiment('test2', 'python super_rl2.py', params.generate_params(randomize=False)),
        ]
        rd = RunDescription('test_run', experiments)
        cmds = rd.generate_experiments()
        for command, name, root_dir in cmds:
            self.assertIn('--experiment', command)
            self.assertIn('--experiments_root', command)
            self.assertTrue(name.startswith('test'))
            self.assertTrue(root_dir.startswith('test_run'))

    def test_simple_cmd(self):
        logging.disable(logging.INFO)

        echo_params = ParamGrid([
            ('p1', [3.14, 2.71]),
            ('p2', ['a', 'b', 'c']),
            ('p3', list(np.arange(3))),
        ])
        experiments = [
            Experiment('test_echo1', 'echo', echo_params.generate_params(randomize=True)),
            Experiment('test_echo2', 'echo', echo_params.generate_params(randomize=False)),
        ]
        rd = RunDescription('test_run', experiments, max_parallel=10)
        run_many(rd)
        logging.disable(logging.NOTSET)
