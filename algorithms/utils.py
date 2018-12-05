"""
Organize experimental data, models, etc.

"""


import os

from os.path import join


# Filesystem helpers

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def project_root():
    """
    Keep models, parameters and summaries at the root of this project's directory tree.
    :return: full path to the root dir of this project.
    """
    return os.path.dirname(os.path.dirname(__file__))


def experiments_dir():
    return ensure_dir_exists(join(project_root(), 'train_dir'))


def experiment_dir(experiment, experiments_root=None):
    if experiments_root is None:
        experiments_root = experiments_dir()
    else:
        experiments_root = join(experiments_dir(), experiments_root)

    return ensure_dir_exists(join(experiments_root, experiment))


def model_dir(experiment_dir_):
    return ensure_dir_exists(join(experiment_dir_, '.model'))


def summaries_dir(experiment_dir_):
    return ensure_dir_exists(join(experiment_dir_, '.summary'))


# Keeping track of experiments

def get_experiment_name(env_id, name):
    return '{}-{}'.format(env_id, name)
