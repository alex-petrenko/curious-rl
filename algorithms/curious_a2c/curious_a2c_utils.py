from algorithms.arguments import parse_args
from algorithms.baselines.a2c import a2c_utils


# values to use if not specified in the command line
DEFAULT_EXPERIMENT_NAME = 'curious_a2c_v000'


def parse_args_curious_a2c(params_cls):
    return parse_args(a2c_utils.DEFAULT_ENV, DEFAULT_EXPERIMENT_NAME, params_cls)
