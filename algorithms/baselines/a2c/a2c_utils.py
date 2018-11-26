from algorithms.arguments import parse_args


# values to use if not specified in the command line
DEFAULT_EXPERIMENT_NAME = 'a2c_v000'
DEFAULT_ENV = 'doom_basic'


def parse_args_a2c(params_cls):
    return parse_args(DEFAULT_ENV, DEFAULT_EXPERIMENT_NAME, params_cls)
