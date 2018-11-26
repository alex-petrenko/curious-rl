import cv2
import sys

from algorithms.curious_a2c.agent_curious_a2c import AgentCuriousA2C
from algorithms.curious_a2c.curious_a2c_utils import parse_args_curious_a2c
from utils.envs.envs import create_env
from utils.utils import log

from algorithms.exploit import run_policy_loop


def enjoy(params, env_id, max_num_episodes=1000000, fps=10):
    def make_env_func():
        e = create_env(env_id, mode='test')
        e.seed(0)
        return e

    agent = AgentCuriousA2C(make_env_func, params.load())
    env = make_env_func()

    # this helps with screen recording
    pause_at_the_beginning = False
    if pause_at_the_beginning:
        env.render()
        log.info('Press any key to start...')
        cv2.waitKey()

    return run_policy_loop(agent, env, max_num_episodes, fps=1000, deterministic=False)


def main():
    args, params = parse_args_curious_a2c(AgentCuriousA2C.Params)
    return enjoy(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
