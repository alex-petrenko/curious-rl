"""
Implementation of the curious variant of the Advantage Actor-Critic algorithm.

"""
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from algorithms.algo_utils import RunningMeanStd, EPS
from algorithms.baselines.a2c.agent_a2c import AgentA2C, Policy
from algorithms.env_wrappers import has_image_observations
from algorithms.multi_env import MultiEnv
from algorithms.tf_utils import dense, count_total_parameters, conv
from algorithms.utils import summaries_dir
from utils.utils import log, AttrDict


class Model:
    """Single class for inverse and forward dynamics model."""

    def __init__(self, env, obs, actions, past_frames, forward_fc):
        """
        :param obs - placeholder for observations
        :param actions - placeholder for selected actions
        """

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-10)

        image_obs = has_image_observations(env)
        obs_shape = list(env.observation_space.shape)
        num_actions = env.action_space.n

        # process observations
        input_shape = [None] + obs_shape  # add batch dimension
        self.observations = obs
        self.next_obs = tf.placeholder(tf.float32, shape=input_shape)

        if image_obs:
            # convolutions
            conv_encoder = tf.make_template(
                'conv_encoder',
                self._convnet_simple,
                create_scope_now_=True,
                convs=[(32, 3, 2)] * 4,
            )
            encoded_obs = conv_encoder(obs=self.observations)
            encoded_obs = tf.contrib.layers.flatten(encoded_obs)

            encoded_next_obs = conv_encoder(obs=self.next_obs)
            self.encoded_next_obs = tf.contrib.layers.flatten(encoded_next_obs)
        else:
            # low-dimensional input
            lowdim_encoder = tf.make_template(
                'lowdim_encoder',
                self._lowdim_encoder,
                create_scope_now_=True,
                past_frames=past_frames,
            )
            encoded_obs = lowdim_encoder(obs=self.observations)
            self.encoded_next_obs = lowdim_encoder(obs=self.next_obs)

        feature_vector_size = encoded_obs.get_shape().as_list()[-1]
        log.info('Feature vector size in ICM: %d', feature_vector_size)

        actions_one_hot = tf.one_hot(actions, num_actions)

        # forward model
        forward_model_input = tf.concat(
            [tf.stop_gradient(encoded_obs), actions_one_hot],  # do not backpropagate to encoder!
            axis=1,
        )
        forward_model_hidden = dense(forward_model_input, forward_fc, self.regularizer)
        forward_model_hidden = dense(forward_model_hidden, forward_fc, self.regularizer)
        forward_model_output = tf.contrib.layers.fully_connected(
            forward_model_hidden, feature_vector_size, activation_fn=None,
        )
        self.predicted_obs = forward_model_output

        # inverse model
        inverse_model_input = tf.concat([encoded_obs, self.encoded_next_obs], axis=1)
        inverse_model_hidden = dense(inverse_model_input, 256, self.regularizer)
        inverse_model_output = tf.contrib.layers.fully_connected(
            inverse_model_hidden, num_actions, activation_fn=None,
        )
        self.predicted_actions = inverse_model_output

        log.info('Total parameters in the model: %d', count_total_parameters())

    def _fc_frame_encoder(self, x):
        return dense(x, 128, self.regularizer)

    def _lowdim_encoder(self, obs, past_frames):
        frames = tf.split(obs, past_frames, axis=1)
        fc_encoder = tf.make_template('fc_encoder', self._fc_frame_encoder, create_scope_now_=True)
        encoded_frames = [fc_encoder(frame) for frame in frames]
        encoded_input = tf.concat(encoded_frames, axis=1)
        return encoded_input

    def _conv(self, x, filters, kernel, stride, scope=None):
        return conv(x, filters, kernel, stride=stride, regularizer=self.regularizer, scope=scope)

    def _convnet_simple(self, convs, obs):
        """Basic stacked convnet."""
        layer = obs
        layer_idx = 1
        for filters, kernel, stride in convs:
            layer = self._conv(layer, filters, kernel, stride, 'conv' + str(layer_idx))
            layer_idx += 1
        return layer


class AgentCuriousA2C(AgentA2C):
    """Agent based on A2C algorithm."""

    class Params(AgentA2C.Params):
        """Hyperparams for the algorithm and the training process."""

        def __init__(self, experiment_name):
            """Default parameter values set in ctor."""
            super(AgentCuriousA2C.Params, self).__init__(experiment_name)
            self.icm_beta = 0.99  # in ICM, importance of training forward model vs inverse model
            self.model_lr_scale = 100.0  # in ICM, importance of model loss vs actor-critic loss
            self.prediction_bonus_coeff = 0.02  # scaling factor for prediction bonus vs env rewards

            self.clip_bonus = 0.05
            self.clip_advantage = 5
            self.normalize_rewards = False

            self.forward_fc = 512

        # noinspection PyMethodMayBeStatic
        def filename_prefix(self):
            return 'curious_a2c_'

    def __init__(self, make_env_func, params):
        """Initialize A2C computation graph and some auxiliary tensors."""
        super(AgentA2C, self).__init__(params)  # calling grandparent ctor, skipping parent

        global_step = tf.train.get_or_create_global_step()

        self.make_env_func = make_env_func

        self.selected_actions = tf.placeholder(tf.int32, [None])  # action selected by the policy
        self.value_estimates = tf.placeholder(tf.float32, [None])
        self.discounted_rewards = tf.placeholder(tf.float32, [None])  # estimate of total reward (rollout + value)
        self.advantages = tf.placeholder(tf.float32, [None])

        env = make_env_func()  # we need it to query observation shape, number of actions, etc.
        self.policy = Policy(
            env,
            params.image_model_name,
            params.fc_layers,
            params.fc_size,
            params.lowdim_model_name,
            params.stack_past_frames,
        )

        self.model = Model(
            env, self.policy.observations, self.selected_actions, params.stack_past_frames, params.forward_fc,
        )

        env.close()

        # negative logarithm of the probabilities of actions
        neglogp_actions = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.policy.actions, labels=self.selected_actions,
        )

        # maximize probabilities of actions that give high advantage
        action_losses = tf.clip_by_value(self.advantages * neglogp_actions, -20.0, 20.0)
        action_loss = tf.reduce_mean(action_losses)

        # penalize for inaccurate value estimation
        value_losses = tf.square(self.discounted_rewards - self.policy.value)
        value_losses = tf.clip_by_value(value_losses, -20.0, 20.0)
        value_loss = self.params.value_loss_coeff * tf.reduce_mean(value_losses)

        # penalize the agent for being "too sure" about it's actions (to prevent converging to the suboptimal local
        # minimum too soon)
        entropy_loss = -tf.reduce_mean(self.policy.actions_prob_distribution.entropy())

        entropy_loss_coeff = tf.train.exponential_decay(
            self.params.initial_entropy_loss_coeff, tf.cast(global_step, tf.float32), 20.0, 0.95, staircase=True,
        )
        entropy_loss_coeff = tf.maximum(entropy_loss_coeff, self.params.min_entropy_loss_coeff)
        entropy_loss = entropy_loss_coeff * entropy_loss

        # total actor-critic loss
        a2c_loss = action_loss + entropy_loss + value_loss

        # model losses
        forward_loss_batch = tf.square(
            tf.stop_gradient(self.model.encoded_next_obs) - self.model.predicted_obs,  # do not backprop to encoder!
        )
        forward_loss_batch = tf.reduce_mean(forward_loss_batch, axis=1)
        forward_loss = tf.reduce_mean(forward_loss_batch)

        inverse_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.model.predicted_actions, labels=self.selected_actions,
        ))
        model_loss = forward_loss * self.params.icm_beta + inverse_loss * (1 - self.params.icm_beta)
        model_loss = self.params.model_lr_scale * model_loss

        # regularization
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # total loss
        loss = a2c_loss + model_loss + regularization_loss

        # training
        self.train = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=self.params.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=self.params.clip_gradients,
        )

        bonus = self.params.prediction_bonus_coeff * forward_loss_batch
        self.prediction_curiosity_bonus = tf.clip_by_value(bonus, -self.params.clip_bonus, self.params.clip_bonus)

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

            tf.summary.scalar('disc_rewards_avg', tf.reduce_mean(self.discounted_rewards))
            tf.summary.scalar('disc_rewards_max', tf.reduce_max(self.discounted_rewards))
            tf.summary.scalar('disc_rewards_min', tf.reduce_min(self.discounted_rewards))

            tf.summary.scalar('bonus_avg', tf.reduce_mean(self.prediction_curiosity_bonus))
            tf.summary.scalar('bonus_max', tf.reduce_max(self.prediction_curiosity_bonus))
            tf.summary.scalar('bonus_min', tf.reduce_min(self.prediction_curiosity_bonus))

            tf.summary.scalar('value', tf.reduce_mean(self.policy.value))

            tf.summary.scalar('adv_avg_abs', tf.reduce_mean(tf.abs(self.advantages)))
            tf.summary.scalar('adv_max', tf.reduce_max(self.advantages))
            tf.summary.scalar('adv_min', tf.reduce_min(self.advantages))

            tf.summary.scalar('selected_action_avg', tf.reduce_mean(tf.to_float(self.selected_actions)))

            tf.summary.scalar('policy_entropy', tf.reduce_mean(self.policy.actions_prob_distribution.entropy()))
            tf.summary.scalar('entropy_coeff', entropy_loss_coeff)

        with tf.name_scope('a2c_losses'):
            tf.summary.scalar('action_loss', action_loss)
            tf.summary.scalar('max_action_loss', tf.reduce_max(action_losses))

            tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('max_value_loss', tf.reduce_max(value_losses))

            tf.summary.scalar('entropy_loss', entropy_loss)
            tf.summary.scalar('a2c_loss', a2c_loss)

            tf.summary.scalar('forward_loss', forward_loss)
            tf.summary.scalar('inverse_loss', inverse_loss)
            tf.summary.scalar('model_loss', model_loss)

            tf.summary.scalar('regularization_loss', regularization_loss)

            tf.summary.scalar('loss', loss)

            summary_dir = summaries_dir(self.params.experiment_name())
            self.summary_writer = tf.summary.FileWriter(summary_dir)

            self.all_summaries = tf.summary.merge_all()

        with tf.name_scope('a2c_aux_summary'):
            tf.summary.scalar('training_steps', global_step, collections=['aux'])

            # if it's not "initialized" yet, just report 0 to preserve tensorboard plot scale
            best_reward_report = tf.cond(
                tf.equal(self.best_avg_reward, self.initial_best_avg_reward),
                true_fn=lambda: 0.0,
                false_fn=lambda: self.best_avg_reward,
            )
            tf.summary.scalar('best_reward_ever', best_reward_report, collections=['aux'])
            tf.summary.scalar('avg_reward', self.avg_reward_placeholder, collections=['aux'])

            self.avg_length_placeholder = tf.placeholder(tf.float32, [])
            tf.summary.scalar('avg_lenght', self.avg_length_placeholder, collections=['aux'])

            self.aux_summaries = tf.summary.merge_all(key='aux')

        self.saver = tf.train.Saver(max_to_keep=3)

        all_vars = tf.trainable_variables()
        log.warn('curious a2c variables:')
        slim.model_analyzer.analyze_vars(all_vars, print_info=True)

    def _prediction_curiosity_bonus(self, observations, actions, next_obs):
        bonuses = self.session.run(
            self.prediction_curiosity_bonus,
            feed_dict={
                self.policy.observations: observations,
                self.selected_actions: actions,
                self.model.next_obs: next_obs,
            }
        )
        return bonuses

    def _curious_train_step(
            self, step, env_steps, observations, actions, values, discounted_rewards, advantages, next_obs
    ):
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
                self.advantages: advantages,
                self.model.next_obs: next_obs,
            },
        )

        step = tf.train.global_step(self.session, tf.train.get_global_step())
        if with_summaries:
            summary = result[1]
            self.summary_writer.add_summary(summary, global_step=env_steps)

        return step

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

        rew_running_mean_std = RunningMeanStd(max_past_samples=100000)
        adv_running_mean_std = RunningMeanStd(max_past_samples=10000)

        def end_of_training(s): return s >= self.params.train_for_steps

        while not end_of_training(step):
            timing = AttrDict({'experience': time.time(), 'batch': time.time()})
            experience_start = time.time()

            env_steps_before_batch = env_steps
            batch_obs = [observations]
            env_steps += len(observations)
            batch_actions, batch_values, batch_rewards, batch_dones, batch_next_obs = [], [], [], [], []
            terminated_by_timer = []
            for rollout_step in range(self.params.rollout):
                actions, values = self._policy_step(observations)
                batch_actions.append(actions)
                batch_values.append(values)

                # wait for all the workers to complete an environment step
                next_obs, rewards, dones, infos = multi_env.step(actions)

                # calculate curiosity bonus
                bonuses = self._prediction_curiosity_bonus(observations, actions, next_obs)
                rewards += bonuses

                # normalize rewards, dividing by the running estimate of standard deviation
                if self.params.normalize_rewards:
                    rew_running_mean_std.update(rewards)
                    rewards /= (np.sqrt(rew_running_mean_std.var) + EPS)

                batch_rewards.append(rewards)
                batch_dones.append(dones)
                batch_next_obs.append(next_obs)
                terminated_by_timer.append(
                    ['terminated_by_timer' in info and info['terminated_by_timer'] for info in infos]
                )

                observations = next_obs

                if infos is not None and 'num_frames' in infos[0]:
                    env_steps += sum((info['num_frames'] for info in infos))
                else:
                    env_steps += multi_env.num_envs

                if rollout_step != self.params.rollout - 1:
                    # we don't need the newest observation in the training batch, already have enough
                    batch_obs.append(observations)

            assert len(batch_obs) == len(batch_rewards)
            assert len(batch_obs) == len(batch_next_obs)

            batch_rewards = np.asarray(batch_rewards, np.float32).swapaxes(0, 1)
            batch_dones = np.asarray(batch_dones, np.bool).swapaxes(0, 1)
            terminated_by_timer = np.asarray(terminated_by_timer, np.bool).swapaxes(0, 1)
            batch_values = np.asarray(batch_values, np.float32).swapaxes(0, 1)

            # Last value won't be valid for envs with done=True (because env automatically resets and shows 1st
            # observation of the next episode. But that's okay, because we should never use last_value in this case.
            last_values = self._estimate_values(observations)

            gamma = self.params.gamma
            disc_rewards = []
            for i in range(len(batch_rewards)):
                env_rewards = self._calc_discounted_rewards(
                    gamma,
                    batch_rewards[i],
                    batch_dones[i],
                    terminated_by_timer[i],
                    batch_values[i],
                    last_values[i],
                )
                disc_rewards.extend(env_rewards)
            disc_rewards = np.asarray(disc_rewards, np.float32)

            # convert observations and estimations to meaningful n-step batches
            batch_obs_shape = (self.params.rollout * multi_env.num_envs, ) + observations[0].shape
            batch_obs = np.asarray(batch_obs, np.float32).swapaxes(0, 1).reshape(batch_obs_shape)
            batch_next_obs = np.asarray(batch_next_obs, np.float32).swapaxes(0, 1).reshape(batch_obs_shape)
            batch_actions = np.asarray(batch_actions, np.int32).swapaxes(0, 1).flatten()
            batch_values = batch_values.flatten()

            advantages = disc_rewards - batch_values
            if self.params.normalize_adv:
                adv_running_mean_std.update(advantages)
                advantages = (advantages - adv_running_mean_std.mean) / (np.sqrt(adv_running_mean_std.var) + EPS)
            advantages = np.clip(advantages, -self.params.clip_advantage, self.params.clip_advantage)

            timing.experience = time.time() - timing.experience
            timing.train = time.time()

            step = self._curious_train_step(
                step, env_steps, batch_obs, batch_actions, batch_values, disc_rewards, advantages, batch_next_obs,
            )
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
