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


def experiment_dir(experiment):
    return ensure_dir_exists(join(experiments_dir(), experiment))


def model_dir(experiment):
    return ensure_dir_exists(join(experiment_dir(experiment), '.model'))


def stats_dir(experiment):
    return ensure_dir_exists(join(experiment_dir(experiment), '.stats'))


def summaries_dir(experiment):
    return ensure_dir_exists(join(experiment_dir(experiment), '.summary'))


def vis_dir(experiment):
    return ensure_dir_exists(join(experiment_dir(experiment), '.vis'))


# Keeping track of experiments

def get_experiment_name(env_id, name):
    return '{}-{}'.format(env_id, name)
