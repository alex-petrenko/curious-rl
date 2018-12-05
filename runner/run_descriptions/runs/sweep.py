from runner.run_descriptions.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('icm_beta', [0.2, 0.5, 0.8]),
    ('prediction_bonus_coeff', [0.01, 0.02, 0.05]),
])

_experiment = Experiment(
    'doom_sweep',
    'python -m algorithms.curious_a2c.train_curious_a2c --env=doom_maze_very_sparse --gpu_mem_fraction=0.1 --train_for_env_steps=80000000',
    _params.generate_params(randomize=False),
)

DOOM_SWEEP_RUN = RunDescription('doom_sweep', experiments=[_experiment], max_parallel=5)
