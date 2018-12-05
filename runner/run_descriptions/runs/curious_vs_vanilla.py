from runner.run_descriptions.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('prediction_bonus_coeff', [0.00, 0.05]),
])

_experiments = [
    Experiment(
        'doom_maze_very_sparse',
        'python -m algorithms.curious_a2c.train_curious_a2c --env=doom_maze_very_sparse --gpu_mem_fraction=0.1 --train_for_env_steps=100000000',
        _params.generate_params(randomize=False),
    ),
    Experiment(
        'doom_maze_sparse',
        'python -m algorithms.curious_a2c.train_curious_a2c --env=doom_maze_sparse --gpu_mem_fraction=0.1 --train_for_env_steps=100000000',
        _params.generate_params(randomize=False),
    ),
    Experiment(
        'doom_maze',
        'python -m algorithms.curious_a2c.train_curious_a2c --env=doom_maze --gpu_mem_fraction=0.1 --train_for_env_steps=50000000',
        _params.generate_params(randomize=False),
    ),
    Experiment(
        'doom_basic',
        'python -m algorithms.curious_a2c.train_curious_a2c --env=doom_basic --gpu_mem_fraction=0.1 --train_for_env_steps=10000000',
        _params.generate_params(randomize=False),
    ),
]

DOOM_CURIOUS_VS_VANILLA = RunDescription('doom_curious_vs_vanilla', experiments=_experiments, max_parallel=5)
