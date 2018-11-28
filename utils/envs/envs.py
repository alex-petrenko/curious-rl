import gym

from algorithms.env_wrappers import RewardScalingWrapper, TimeLimitWrapper, SkipAndStackFramesWrapper
from utils.doom.doom_utils import make_doom_env, env_by_name


def create_env(env, **kwargs):
    if env.startswith('doom_'):
        mode = '_'.join(env.split('_')[1:])
        return make_doom_env(env_by_name(mode), **kwargs)
    elif env.startswith('car'):
        env = gym.make('MountainCar-v0').unwrapped
        env = TimeLimitWrapper(env, limit=200, random_variation_steps=1)
        env = SkipAndStackFramesWrapper(env, num_frames=3)
        env = RewardScalingWrapper(env, 0.1)
        return env
    else:
        raise Exception('Unsupported env {0}'.format(env))
