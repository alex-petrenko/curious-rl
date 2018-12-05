from runner.run_descriptions.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('icm_beta', [0.2]),
])

_experiments = [
    Experiment(
        'doom_test_basic',
        'python -m algorithms.curious_a2c.train_curious_a2c --env=doom_basic --gpu_mem_fraction=0.1 --train_for_env_steps=80000',
        _params.generate_params(randomize=False),
    ),
    Experiment(
        'doom_test_maze',
        'python -m algorithms.curious_a2c.train_curious_a2c --env=doom_maze --gpu_mem_fraction=0.1 --train_for_env_steps=80000',
        _params.generate_params(randomize=False),
    ),
]

DOOM_TEST_RUN = RunDescription('doom_test', experiments=_experiments, max_parallel=3)
