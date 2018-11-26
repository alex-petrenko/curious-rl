from utils.doom.doom_utils import make_doom_env, env_by_name


def create_env(env, **kwargs):
    if env.startswith('doom_'):
        mode = '_'.join(env.split('_')[1:])
        return make_doom_env(env_by_name(mode), **kwargs)
    else:
        raise Exception('Unsupported env {0}'.format(env))
