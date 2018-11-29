"""
Implementation of the synchronous variant of the Advantage Actor-Critic algorithm.

"""


import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from algorithms.algo_utils import EPS
from algorithms.env_wrappers import has_image_observations
from algorithms.multi_env import MultiEnv
from utils.utils import log, put_kernels_on_grid, AttrDict

from algorithms.utils import *
from algorithms.agent import AgentLearner
from algorithms.tf_utils import count_total_parameters, dense, conv

from utils.distributions import CategoricalProbabilityDistribution


class Policy:
    """A class that represents both the actor's policy and the value estimator."""

    def __init__(self, env, img_model_name, fc_layers, fc_size, lowdim_model_name, past_frames):
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-10)

        image_obs = has_image_observations(env)
        obs_shape = list(env.observation_space.shape)
        num_actions = env.action_space.n

        # process observations
        input_shape = [None] + obs_shape  # add batch dimension
        self.observations = tf.placeholder(tf.float32, shape=input_shape)

        if image_obs:
            # convolutions
            if img_model_name == 'convnet_simple':
                conv_filters = self._convnet_simple([(32, 3, 2)] * 4)
            else:
                raise Exception('Unknown model name')

            encoded_input = tf.contrib.layers.flatten(conv_filters)
        else:
            # low-dimensional input
            if lowdim_model_name == 'simple_fc':
                frames = tf.split(self.observations, past_frames, axis=1)
                fc_encoder = tf.make_template('fc_encoder', self._fc_frame_encoder, create_scope_now_=True)
                encoded_frames = [fc_encoder(frame) for frame in frames]
                encoded_input = tf.concat(encoded_frames, axis=1)
            else:
                raise Exception('Unknown lowdim model name')

        fc = encoded_input
        for _ in range(fc_layers - 1):
            fc = dense(fc, fc_size, self.regularizer)

        # fully-connected layers to generate actions
        actions_fc = dense(fc, fc_size // 2, self.regularizer)
        self.actions = tf.contrib.layers.fully_connected(actions_fc, num_actions, activation_fn=None)
        self.best_action_deterministic = tf.argmax(self.actions, axis=1)
        self.actions_prob_distribution = CategoricalProbabilityDistribution(self.actions)
        self.act = self.actions_prob_distribution.sample()

        value_fc = dense(fc, fc_size // 2, self.regularizer)
        self.value = tf.squeeze(tf.contrib.layers.fully_connected(value_fc, 1, activation_fn=None), axis=[1])

        if image_obs:
            # summaries
            with tf.variable_scope('conv1', reuse=True):
                weights = tf.get_variable('weights')
            with tf.name_scope('a2c_agent_summary_conv'):
                if weights.shape[2].value in [1, 3, 4]:
                    tf.summary.image('conv1/kernels', put_kernels_on_grid(weights), max_outputs=1)

        log.info('Total parameters in the model: %d', count_total_parameters())

    def _fc_frame_encoder(self, x):
        return dense(x, 128, self.regularizer)

    def _conv(self, x, filters, kernel, stride, scope=None):
        return conv(x, filters, kernel, stride=stride, regularizer=self.regularizer, scope=scope)

    def _convnet_simple(self, convs):
        """Basic stacked convnet."""
        layer = self.observations
        layer_idx = 1
        for filters, kernel, stride in convs:
            layer = self._conv(layer, filters, kernel, stride, 'conv' + str(layer_idx))
            layer_idx += 1
        return layer


class AgentA2C(AgentLearner):
    """Agent based on A2C algorithm."""

    class Params(AgentLearner.AgentParams):
        """Hyperparams for the algorithm and the training process."""

        def __init__(self, experiment_name):
            """Default parameter values set in ctor."""
            super(AgentA2C.Params, self).__init__(experiment_name)

            self.gamma = 0.99  # future reward discount
            self.rollout = 5  # number of successive env steps used for each model update
            self.num_envs = 40  # number of environments running in parallel. Batch size = rollout * num_envs
            self.num_workers = 20  # number of workers used to run the environments

            self.stack_past_frames = 3
            self.num_input_frames = self.stack_past_frames

            # policy
            self.image_model_name = 'convnet_simple'
            self.fc_layers = 2
            self.fc_size = 256
            self.lowdim_model_name = 'simple_fc'

            # components of the loss function
            self.initial_entropy_loss_coeff = 0.1
            self.min_entropy_loss_coeff = 0.002
            self.value_loss_coeff = 1.0

            # training process
            self.normalize_adv = False
            self.learning_rate = 1e-4
            self.clip_gradients = 20.0
            self.print_every = 50
            self.train_for_steps = 5000000
            self.use_gpu = True

        # noinspection PyMethodMayBeStatic
        def filename_prefix(self):
            return 'a2c_'

    def __init__(self, make_env_func, params):
        """Initialize A2C computation graph and some auxiliary tensors."""
        super(AgentA2C, self).__init__(params)

        global_step = tf.train.get_or_create_global_step()

        self.make_env_func = make_env_func

        env = make_env_func()  # we need it to query observation shape, number of actions, etc.
        self.policy = Policy(
            env,
            params.image_model_name,
            params.fc_layers,
            params.fc_size,
            params.lowdim_model_name,
            params.stack_past_frames,
        )
        env.close()

        self.selected_actions = tf.placeholder(tf.int32, [None])  # action selected by the policy
        self.value_estimates = tf.placeholder(tf.float32, [None])
        self.discounted_rewards = tf.placeholder(tf.float32, [None])  # estimate of total reward (rollout + value)

        advantages = self.discounted_rewards - self.value_estimates
        if self.params.normalize_adv:
            advantages = advantages / tf.reduce_max(tf.abs(advantages))  # that's a crude way

        # negative logarithm of the probabilities of actions
        neglogp_actions = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.policy.actions, labels=self.selected_actions,
        )

        # maximize probabilities of actions that give high advantage
        action_loss = tf.reduce_mean(tf.clip_by_value(advantages * neglogp_actions, -20.0, 20.0))

        # penalize for inaccurate value estimation
        value_loss = tf.losses.mean_squared_error(self.discounted_rewards, self.policy.value)
        value_loss = self.params.value_loss_coeff * value_loss

        # penalize the agent for being "too sure" about it's actions (to prevent converging to the suboptimal local
        # minimum too soon)
        entropy_loss = -tf.reduce_mean(self.policy.actions_prob_distribution.entropy())

        entropy_loss_coeff = tf.train.exponential_decay(
            self.params.initial_entropy_loss_coeff, tf.cast(global_step, tf.float32), 20.0, 0.95, staircase=True,
        )
        entropy_loss_coeff = tf.maximum(entropy_loss_coeff, self.params.min_entropy_loss_coeff)
        entropy_loss = entropy_loss_coeff * entropy_loss

        a2c_loss = action_loss + entropy_loss + value_loss
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = regularization_loss + a2c_loss

        # training
        self.train = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=self.params.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=self.params.clip_gradients,
        )

        # summaries for the agent and the training process
        with tf.name_scope('a2c_agent_summary'):
            if len(self.policy.observations.shape) >= 4:
                tf.summary.image(
                    'observations',
                    self.policy.observations[:, :, :, :3],  # first three channels
                    max_outputs=8,
                )
                # output also last channel
                if self.policy.observations.shape[-1].value > 4:
                    tf.summary.image('observations_last_channel', self.policy.observations[:, :, :, -1:])

            tf.summary.scalar('value', tf.reduce_mean(self.policy.value))
            tf.summary.scalar('avg_abs_advantage', tf.reduce_mean(tf.abs(advantages)))

            # tf.summary.histogram('actions', self.policy.actions)
            tf.summary.scalar('action_avg', tf.reduce_mean(tf.to_float(self.policy.act)))

            # tf.summary.histogram('selected_actions', self.selected_actions)
            tf.summary.scalar('selected_action_avg', tf.reduce_mean(tf.to_float(self.selected_actions)))

            tf.summary.scalar('policy_entropy', tf.reduce_mean(self.policy.actions_prob_distribution.entropy()))
            tf.summary.scalar('entropy_coeff', entropy_loss_coeff)

            tf.summary.scalar('action_loss', action_loss)
            tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('entropy_loss', entropy_loss)
            tf.summary.scalar('a2c_loss', a2c_loss)
            tf.summary.scalar('regularization_loss', regularization_loss)
            tf.summary.scalar('loss', loss)

            summary_dir = summaries_dir(self.params.experiment_name())
            self.summary_writer = tf.summary.FileWriter(summary_dir)

            self.all_summaries = tf.summary.merge_all()

        with tf.name_scope('a2c_aux_summary'):
            tf.summary.scalar('training_steps', global_step, collections=['aux'])
            tf.summary.scalar('best_reward_ever', self.best_avg_reward, collections=['aux'])
            tf.summary.scalar('avg_reward', self.avg_reward_placeholder, collections=['aux'])

            self.avg_length_placeholder = tf.placeholder(tf.float32, [])
            tf.summary.scalar('avg_lenght', self.avg_length_placeholder, collections=['aux'])

            self.aux_summaries = tf.summary.merge_all(key='aux')

        self.saver = tf.train.Saver(max_to_keep=3)

        all_vars = tf.trainable_variables()
        log.warn('a2c variables:')
        slim.model_analyzer.analyze_vars(all_vars, print_info=True)

    def finalize(self):
        super(AgentA2C, self).finalize()

    def _maybe_print(self, step, avg_rewards, avg_length, fps, t):
        if step % self.params.print_every == 0:
            log.info('<====== Step %d ======>', step)
            log.info('Avg FPS: %.1f', fps)
            log.info('Experience for batch took %.3f sec (%.1f batches/s)', t.experience, 1.0 / t.experience)
            log.info('Train step for batch took %.3f sec (%.1f batches/s)', t.train, 1.0 / t.train)
            log.info('Avg. %d episode lenght: %.3f', self.params.stats_episodes, avg_length)

            best_avg_reward = self.best_avg_reward.eval(session=self.session)
            log.info(
                'Avg. %d episode reward: %.3f (best: %.3f)',
                self.params.stats_episodes, avg_rewards, best_avg_reward,
            )

    def _maybe_aux_summaries(self, step, env_steps, avg_reward, avg_length):
        if self._should_write_summaries(step):
            summary = self.session.run(
                self.aux_summaries,
                feed_dict={
                    self.avg_reward_placeholder: avg_reward,
                    self.avg_length_placeholder: avg_length,
                },
            )
            self.summary_writer.add_summary(summary, global_step=env_steps)

    def best_action(self, observation, deterministic=False):
        actions, _ = self._policy_step([observation], deterministic)
        return actions[0]

    def _policy_step(self, observations, deterministic=False):
        """
        Select the best action by sampling from the distribution generated by the policy. Also estimate the
        value for the currently observed environment state.
        """
        ops = [
            self.policy.best_action_deterministic if deterministic else self.policy.act,
            self.policy.value,
        ]
        actions, values = self.session.run(ops, feed_dict={self.policy.observations: observations})
        return actions, values

    def _estimate_values(self, observations):
        values = self.session.run(
            self.policy.value,
            feed_dict={self.policy.observations: observations},
        )
        return values

    def _train_step(self, step, env_steps, observations, actions, values, discounted_rewards):
        """
        Actually do a single iteration of training. See the computational graph in the ctor to figure out
        the details.
        """
        with_summaries = self._should_write_summaries(step)
        summaries = [self.all_summaries] if with_summaries else []
        result = self.session.run(
            [self.train] + summaries,
            feed_dict={
                self.policy.observations: observations,
                self.selected_actions: actions,
                self.value_estimates: values,
                self.discounted_rewards: discounted_rewards,
            },
        )

        step = tf.train.global_step(self.session, tf.train.get_global_step())
        if with_summaries:
            summary = result[1]
            self.summary_writer.add_summary(summary, global_step=env_steps)

        return step

    @staticmethod
    def _calc_discounted_rewards(gamma, rewards, dones, last_value):
        """Calculate gamma-discounted rewards for an n-step A2C."""
        cumulative = 0 if dones[-1] else last_value
        discounted_rewards = []
        for rollout_step in reversed(range(len(rewards))):
            r, done = rewards[rollout_step], dones[rollout_step]
            cumulative = r + gamma * cumulative * (not done)
            discounted_rewards.append(cumulative)
        return reversed(discounted_rewards)

    def learn(self, step_callback=None):
        """
        Main training loop.
        :param step_callback: a hacky callback that takes a dictionary with all local variables as an argument.
        Allows you too look inside the training process.
        """
        step = initial_step = tf.train.global_step(self.session, tf.train.get_global_step())
        env_steps = env_steps_initial = self.total_env_steps.eval(session=self.session)
        batch_size = self.params.rollout * self.params.num_envs

        multi_env = MultiEnv(
            self.params.num_envs,
            self.params.num_workers,
            make_env_func=self.make_env_func,
            stats_episodes=self.params.stats_episodes,
        )
        observations = multi_env.initial_obs()

        def end_of_training(s): return s >= self.params.train_for_steps

        while not end_of_training(step):
            timing = AttrDict({'experience': time.time(), 'batch': time.time()})
            experience_start = time.time()

            env_steps_before_batch = env_steps
            batch_obs = [observations]
            env_steps += len(observations)
            batch_actions, batch_values, batch_rewards, batch_dones = [], [], [], []
            for rollout_step in range(self.params.rollout):
                actions, values = self._policy_step(observations)
                batch_actions.append(actions)
                batch_values.append(values)

                # wait for all the workers to complete an environment step
                observations, rewards, dones, infos = multi_env.step(actions)
                batch_rewards.append(rewards)
                batch_dones.append(dones)
                if infos is not None and 'num_frames' in infos[0]:
                    env_steps += sum((info['num_frames'] for info in infos))
                else:
                    env_steps += multi_env.num_envs

                if rollout_step != self.params.rollout - 1:
                    # we don't need the newest observation in the training batch, already have enough
                    batch_obs.append(observations)

            assert len(batch_obs) == len(batch_rewards)

            batch_rewards = np.asarray(batch_rewards, np.float32).swapaxes(0, 1)
            batch_dones = np.asarray(batch_dones, np.bool).swapaxes(0, 1)
            last_values = self._estimate_values(observations)

            gamma = self.params.gamma
            discounted_rewards = []
            for env_rewards, env_dones, last_value in zip(batch_rewards, batch_dones, last_values):
                discounted_rewards.extend(self._calc_discounted_rewards(gamma, env_rewards, env_dones, last_value))

            # convert observations and estimations to meaningful n-step batches
            batch_obs_shape = (self.params.rollout * multi_env.num_envs, ) + observations[0].shape
            batch_obs = np.asarray(batch_obs, np.float32).swapaxes(0, 1).reshape(batch_obs_shape)
            batch_actions = np.asarray(batch_actions, np.int32).swapaxes(0, 1).flatten()
            batch_values = np.asarray(batch_values, np.float32).swapaxes(0, 1).flatten()

            timing.experience = time.time() - timing.experience
            timing.train = time.time()

            step = self._train_step(step, env_steps, batch_obs, batch_actions, batch_values, discounted_rewards)
            self._maybe_save(step, env_steps)

            timing.train = time.time() - timing.train

            avg_reward = multi_env.calc_avg_rewards(n=self.params.stats_episodes)
            avg_length = multi_env.calc_avg_episode_lengths(n=self.params.stats_episodes)
            fps = (env_steps - env_steps_before_batch) / (time.time() - timing.batch)

            self._maybe_print(step, avg_reward, avg_length, fps, timing)
            self._maybe_aux_summaries(step, env_steps, avg_reward, avg_length)
            self._maybe_update_avg_reward(avg_reward, env_steps - env_steps_initial)

            if step_callback is not None:
                step_callback(locals(), globals())

        log.info('Done!')
        multi_env.close()
