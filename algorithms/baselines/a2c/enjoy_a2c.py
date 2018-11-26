import cv2
import sys

from utils.envs.envs import create_env
from utils.utils import log

from algorithms.exploit import run_policy_loop

from algorithms.baselines.a2c.a2c_utils import *
from algorithms.baselines.a2c.agent_a2c import AgentA2C


def enjoy(params, env_id, max_num_episodes=1000000, fps=10):
    def make_env_func():
        e = create_env(env_id, mode='test')
        e.seed(0)
        return e

    agent = AgentA2C(make_env_func, params.load())
    env = make_env_func()

    # this helps with screen recording
    pause_at_the_beginning = False
    if pause_at_the_beginning:
        env.render()
        log.info('Press any key to start...')
        cv2.waitKey()

    return run_policy_loop(agent, env, max_num_episodes, fps=1000, deterministic=False)


def main():
    args, params = parse_args_a2c(AgentA2C.Params)
    return enjoy(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
