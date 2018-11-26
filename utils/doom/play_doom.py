import sys

from utils.envs.envs import create_env


def main():
    env = create_env('doom_maze_very_sparse', mode='test')
    return env.unwrapped.play_human_mode()


if __name__ == '__main__':
    sys.exit(main())
