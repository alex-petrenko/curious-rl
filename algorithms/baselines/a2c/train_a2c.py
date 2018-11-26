import sys

from algorithms.baselines.a2c.a2c_utils import *
from algorithms.baselines.a2c.agent_a2c import AgentA2C
from utils.envs.envs import create_env


def train(a2c_params, env_id):
    def make_env_func():
        return create_env(env_id)

    agent = AgentA2C(make_env_func, params=a2c_params)
    agent.initialize()
    agent.learn()
    agent.finalize()
    return 0


def main():
    """Script entry point."""
    args, params = parse_args_a2c(AgentA2C.Params)
    return train(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
