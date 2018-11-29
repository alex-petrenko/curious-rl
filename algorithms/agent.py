"""
Base classes for RL agent implementations with some boilerplate.

"""

import tensorflow as tf

from algorithms.utils import *

from utils.params import Params

from utils.utils import log
from utils.decay import LinearDecay


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class Agent:
    def __init__(self, params):
        self.params = params

    def initialize(self):
        pass

    def finalize(self):
        pass

    def analyze_observation(self, observation):
        """Default implementation, may be or may not be overridden."""
        return None

    def best_action(self, observation):
        """Must be overridden in derived classes."""
        raise NotImplementedError('Subclasses should implement {}'.format(self.best_action.__name__))


class AgentRandom(Agent):
    def __init__(self, params, env):
        super(AgentRandom, self).__init__(params)
        self.action_space = env.action_space

    def best_action(self, *args, **kwargs):
        return self.action_space.sample()


# noinspection PyAbstractClass
class AgentLearner(Agent):
    class AgentParams(Params):
        def __init__(self, experiment_name):
            super(AgentLearner.AgentParams, self).__init__(experiment_name)
            self.use_gpu = True
            self.gpu_mem_fraction = 1.0

            self.stats_episodes = 100  # how many rewards to average to measure performance

    def __init__(self, params):
        super(AgentLearner, self).__init__(params)
        self.session = None  # actually created in "initialize" method
        self.saver = None

        tf.reset_default_graph()

        self.summary_rate_decay = LinearDecay([(0, 100), (1000000, 2000), (10000000, 10000)], staircase=100)
        self.save_rate_decay = LinearDecay([(0, 100), (1000000, 2000)], staircase=100)

        self.initial_best_avg_reward = tf.constant(-1e3)
        self.best_avg_reward = tf.Variable(self.initial_best_avg_reward)
        self.total_env_steps = tf.Variable(0, dtype=tf.int64)

        def update_best_value(best_value, new_value):
            return tf.assign(best_value, tf.maximum(new_value, best_value))
        self.avg_reward_placeholder = tf.placeholder(tf.float32, [], 'new_avg_reward')
        self.update_best_reward = update_best_value(self.best_avg_reward, self.avg_reward_placeholder)
        self.total_env_steps_placeholder = tf.placeholder(tf.int64, [], 'new_env_steps')
        self.update_env_steps = tf.assign(self.total_env_steps, self.total_env_steps_placeholder)

    def initialize(self):
        """Start the session."""
        gpu_options = tf.GPUOptions()
        if self.params.gpu_mem_fraction != 1.0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params.gpu_mem_fraction)

        config = tf.ConfigProto(
            device_count={'GPU': 100 if self.params.use_gpu else 0},
            gpu_options=gpu_options,
            log_device_placement=False,
        )
        self.session = tf.Session(config=config)
        checkpoint_dir = model_dir(self.params.experiment_name())
        try:
            self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
        except ValueError:
            log.info('Didn\'t find a valid restore point, start from scratch')
            self.session.run(tf.global_variables_initializer())
        log.info('Initialized!')

    def finalize(self):
        self.session.close()

    def _maybe_save(self, step, env_steps):
        self.params.ensure_serialized()
        save_every = self.save_rate_decay.at(step)
        if (step + 1) % save_every == 0:
            log.info('Training step #%d, env steps: %d, saving...', step, env_steps)
            saver_path = model_dir(self.params.experiment_name()) + '/' + self.__class__.__name__
            self.session.run(self.update_env_steps, feed_dict={self.total_env_steps_placeholder: env_steps})
            self.saver.save(self.session, saver_path, global_step=step)

    def _should_write_summaries(self, step):
        summaries_every = self.summary_rate_decay.at(step)
        return (step + 1) % summaries_every == 0

    def _maybe_update_avg_reward(self, avg_reward, stats_num_episodes):
        if stats_num_episodes > self.params.stats_episodes:
            curr_best_reward = self.best_avg_reward.eval(session=self.session)
            if avg_reward > curr_best_reward + 1e-6:
                log.warn('New best reward %.6f (was %.6f)!', avg_reward, curr_best_reward)
                self.session.run(self.update_best_reward, feed_dict={self.avg_reward_placeholder: avg_reward})
