import gym
import vizdoomgym

from algorithms.env_wrappers import ResizeAndGrayscaleWrapper, StackFramesWrapper, RewardScalingWrapper, \
    SkipAndStackFramesWrapper
from utils.doom.wrappers import set_resolution

DOOM_W = DOOM_H = 42


class DoomCfg:
    def __init__(self, name, env_id, reward_scaling):
        self.name = name
        self.env_id = env_id
        self.reward_scaling = reward_scaling


DOOM_ENVS = [
    DoomCfg('basic', 'VizdoomBasic-v0', 0.01),
    DoomCfg('maze', 'VizdoomMyWayHome-v0', 1.0),
    DoomCfg('maze_sparse', 'VizdoomMyWayHomeSparse-v0', 1.0),
    DoomCfg('maze_very_sparse', 'VizdoomMyWayHomeVerySparse-v0', 1.0),
]


def env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Doom env')


def make_doom_env(doom_cfg, mode='train'):
    env = gym.make(doom_cfg.env_id)

    # courtesy of https://github.com/pathak22/noreward-rl/blob/master/src/envs.py
    # and https://github.com/ppaquette/gym-doom
    obwrapper = set_resolution('160x120')
    env = obwrapper(env)

    env = ResizeAndGrayscaleWrapper(env, DOOM_W, DOOM_H)

    if mode == 'test':
        # disable action repeat during test time
        env = StackFramesWrapper(env, stack_past_frames=4)
    else:
        # during training we repeat the last action n times and stack the same number of frames to capture dynamics
        env = SkipAndStackFramesWrapper(env, num_frames=4)

    env = RewardScalingWrapper(env, doom_cfg.reward_scaling)
    return env
