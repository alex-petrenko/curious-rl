from runner.run_descriptions.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('ignore_timer', [True, False]),
])

_experiments = [
    Experiment(
        'doom_maze',
        'python -m algorithms.curious_a2c.train_curious_a2c --env=doom_maze --gpu_mem_fraction=0.1 --train_for_env_steps=80000000',
        _params.generate_params(randomize=False),
    ),
    Experiment(
        'doom_basic',
        'python -m algorithms.curious_a2c.train_curious_a2c --env=doom_basic --gpu_mem_fraction=0.1 --train_for_env_steps=10000000',
        _params.generate_params(randomize=False),
    ),
]

DOOM_TIMER = RunDescription('doom_timer', experiments=_experiments, max_parallel=2)
