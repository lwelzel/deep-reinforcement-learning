#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Actor Critic Class
#
#######################
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.signal import lfilter
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mean_squared_error, SparseCategoricalCrossentropy, CategoricalCrossentropy
from base_agent import BaseAgent
from actor_critic_model import ActorCriticModel
from tqdm import tqdm
from buffer_class import MetaBuffer, PrioBuffer


class ActorCriticAgent(object):
    def __init__(self, state_space, action_space,
                 use_BS=True,
                 use_BLS=True,
                 use_ER=False,
                 use_TES=False,
                 use_AN=False,
                 buffer_type=None,
                 buffer_depth=1000, batch_size=64,
                 max_reward=500,
                 exp_policy='egreedy', epsilon=1., temperature=1., tanh_temp=10.,
                 anneal_method='exponential',
                 decay=0.95, epsilon_min=0.01, temp_min=0.1,
                 learning_rate=5e-3, discount=0.95,
                 hidden_layers_actor=(32, 16), hidden_act_actor='relu',
                 kernel_init_actor="glorot_uniform", actor_output_activation="tanh",  # "softmax", None is ok since we assume logits internally
                 hidden_layers_critic=(32, 16), hidden_act_critic='relu',
                 kernel_init_critic="glorot_uniform", critic_output_activation=None,  # must be None to allow any value
                 name="", id=0,
                 **kwargs):
        super(ActorCriticAgent, self).__init__()

        # ENVIRONMENT
        self.state_space = state_space.shape
        self.state_space_size = self.state_space[0]
        self.action_space = action_space
        self.action_state_size = action_space.n

        # RNG
        self.rng = np.random.default_rng()

        # PARAMETERS
        self.gamma = discount
        self.epsilon = epsilon
        self.tanh_temp = tanh_temp
        self.decay = decay
        # regularization coefficients
        self.value_rc = 0.5
        self.entropy_rc = 1e-4
        self.tes_target_entropy = 0.5
        self.test_target_std = 0.1
        self.tes_alpha = 0.01
        self.tes_window = np.zeros((20, batch_size), dtype=float)
        self._tes_counter = 0
        self.batch_size = batch_size

        # TF functions
        # get a weighted, sparse, cross entropy function to compute cross entropy for loss function
        # We have None as the output of our action_probability (actor) network branch.
        # This means that the output probabilities of out actor are logits
        self.weighted_cce_fn = SparseCategoricalCrossentropy(from_logits=True,
                                                             reduction="auto",
                                                             name="weighted_sparse_CCE_softmax_input")
        self.cce_fn = CategoricalCrossentropy(from_logits=False,
                                              reduction="auto",
                                              name="CCE_softmax_input")


        self.cce_tes = tf.nn.sigmoid_cross_entropy_with_logits

        # IMPORTANT: If the model 'actor_prop_rescale_policy' is not a function that normalizes the output probabilities
        # could be # None or softmax/tanh/sigmoid
        # you need to change the 'from_logits' option in 'SparseCategoricalCrossentropy' to 'True'
        # IMPORTANT 2: the activation
        self.model = ActorCriticModel(self.action_state_size, self.state_space_size,
                                      hidden_layers_actor=hidden_layers_actor, hidden_act_actor=hidden_act_actor,
                                      kernel_init_actor=kernel_init_actor,
                                      hidden_layers_critic=hidden_layers_critic, hidden_act_critic=hidden_act_critic,
                                      kernel_init_critic=kernel_init_critic,
                                      actor_output_activation=actor_output_activation, # softmax/tanh/sigmoid
                                      critic_output_activation=critic_output_activation)

        self.model.compile(optimizer=RMSprop(lr=learning_rate),
                           loss=[self.policy_loss, self.value_loss], )


        # ACTOR SETTINGS
        # bootstrapping
        if use_BS:
            self.get_discounted_rewards = self._get_bs_discounted_rewards
        else:
            self.get_discounted_rewards = self._get_no_bs_discounted_rewards
        # baseline subtraction
        if use_BLS:
            self.base_line_subtraction = self._base_line_subtraction
        else:
            self.base_line_subtraction = self._pass_return
        # experience replay
        if use_ER:
            if buffer_type is None:
                self.buffer = MetaBuffer(buffer_depth, batch_size)
            elif buffer_type == "priority":
                self.buffer = PrioBuffer(buffer_depth, batch_size)
            else:
                raise KeyError("No valid buffer type provided.")
            self.get_trace = self._sample_buffer
            self.sample_transition = self.buffer.update_buffer
        else:
            self.get_trace = self._pass_all
            self.sample_transition = self._pass
        # target entropy scheduling
        if use_TES:
            self.update_target_entropy = self._update_target_entropy
        else:
            self.update_target_entropy = self._pass
        # anneal action selection
        if use_AN:
            self.anneal = self._anneal_tanh_temp
        else:
            self.tanh_temp = 1.
            self.anneal = self._pass





    def select_action(self, s, num_samples=1):
        action_probability, value = self.model(s.reshape(-1, 4))
        action_probability = np.tanh(action_probability / self.tanh_temp)
        action = tf.squeeze(tf.random.categorical(action_probability, num_samples=num_samples))
        return action, value

    def _sample_buffer(self, *args, **kwargs):
        return self.buffer.sample

    def _update_target_entropy(self, cce, expected_return, *args, **kwargs):
        np.put(self.tes_window, np.arange(self._tes_counter * self.batch_size,
                                          (self._tes_counter + 1) * self.batch_size),
               - expected_return * cce, mode="wrap")
        mean = np.mean(self.tes_window)
        std = np.std(self.tes_window)

        self.entropy_rc = self.entropy_rc \
                          - self.entropy_rc * self.tes_alpha * np.logical_and(np.isclose(self.tes_target_entropy, mean),
                                                                              self.test_target_std >= std)

        self._tes_counter += 1

    def _anneal_tanh_temp(self):
        self.tanh_temp = np.clip(self.tanh_temp * self.decay, a_min=1.e-2, a_max=None)

    @staticmethod
    @njit(parallel=False, nogil=True, fastmath=True)
    def _get_bs_discounted_rewards(rewards, dones, gamma, predicted_value=1.):
        """
        Compute discounted rewards using rewards over trace with bootstrapping
        """
        # TODO: https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning
        returns = np.zeros(len(rewards) + 1)
        returns[-1] = predicted_value
        for t in np.flip(np.arange(len(rewards))):
            returns[t] = rewards[t] + gamma * np.roll(returns, -1)[t] * (1 - dones[t])
        return returns[:-1]

    @staticmethod
    @njit(parallel=False, nogil=True, fastmath=True)
    def _get_no_bs_discounted_rewards(rewards, dones, gamma, predicted_value=1.):
        """
        Compute discounted rewards using rewards over trace without bootstrapping.
        Maximum time horizon is the batch size. We do not implement other time horizons at the moment
        """
        returns = np.zeros(len(rewards))
        for t in np.flip(np.arange(len(rewards))):
            returns[t] = rewards[t] + gamma * np.roll(returns, -1)[t] * (1 - dones[t])
        return returns

    @staticmethod
    def _pass_return(arg, *args, **kwargs):
        return arg

    @staticmethod
    def _pass_all(*args, **kwargs):
        return args

    @staticmethod
    def _pass(*args, **kwargs):
        pass

    @staticmethod
    @njit(parallel=False, nogil=True, fastmath=True)
    def _base_line_subtraction(returns, target_values):
        return returns - target_values

    def get_advantage(self, target_values, rewards, dones, next_value):
        returns = self.get_discounted_rewards(rewards, dones, self.gamma, next_value)
        return self.base_line_subtraction(returns, target_values), returns

    @tf.function
    def value_loss(self, returns, value):
        return mean_squared_error(returns, value) * self.value_rc

    @tf.function
    def policy_loss(self, actions_advantage, action_probability):
        actions, advantages = tf.split(actions_advantage, 2, axis=-1)

        policy_loss = self.weighted_cce_fn(actions,
                                           action_probability,
                                           sample_weight=advantages)

        # rescale action probability using softmax
        # TODO: figure out how to use an automatically rescaled output from NN by setting the output activation
        #  I have been trying this but it really didnt work well.
        #  The agent still learns but performs much (50%) worse if I use non-logit probabilities as NN outputs for the actor.
        #  Could we please get feedback on this, because I just dont get it.
        #  You will need to do changes in the following lines:
        #  l 28: in agent settings set actor_output_activation="softmax"
        #  l 54: in self.weighted_cce_fn = SparseCategoricalCrossentropy() set from_logits=False
        #  l 119: comment out line 119 - action_probability = tf.nn.softmax(action_probability)
        action_probability = tf.nn.softmax(action_probability)

        entropy_loss = self.cce_fn(action_probability,
                                   action_probability)

        return policy_loss - entropy_loss * self.entropy_rc

    def update_policy(self, trace_array, actions, rewards, values, dones, next_state):
        __, next_value = self.model(np.reshape(next_state, (-1, self.state_space_size)))

        # TODO: ACER PER sample selection



        advantages, returns = self.get_advantage(target_values=values,
                                                 rewards=rewards,
                                                 dones=dones,
                                                 next_value=float(next_value))

        actions_advantage = np.stack((actions, advantages), axis=-1)

        # TODO: Target Entropy Update
        # wcce = self.cce_tes(actions.astype(float), returns)
        # self.update_target_entropy(wcce, returns)

        losses = self.model.train_on_batch(trace_array,
                                           [actions_advantage,
                                            returns])

        # self.anneal()
        return losses


@njit(parallel=False, nogil=True, fastmath=True)
def adjust_reward(r, i, done, max_steps, adjust_factor=0.1, adjust=True):
    r_incentive = np.logical_and(done, i != max_steps) * adjust
    r = (1 - r_incentive) * r + r_incentive * - adjust_factor * max_steps
    return int(r)


def train_actor_critic(env, num_epochs, batch_size=64, adjust_factor=0.1, adjust=True, **kwargs):
    pi = ActorCriticAgent(env.observation_space, env.action_space, batch_size=batch_size, **kwargs)
    pi.model.summary()
    plot_model(pi.model, "a2c_model_graph.png",
               show_shapes=True,
               show_layer_names=True,
               expand_nested=True,
               )

    actions = np.full(batch_size, fill_value=np.nan, dtype=int)
    rewards, dones, values = np.full((3, batch_size), fill_value=np.nan)
    states = np.full((batch_size, 4), fill_value=np.nan)
    episode_rewards = np.zeros(num_epochs * 2, dtype=int)
    batch_counter = np.arange(batch_size)

    t_ep = 0
    timestep = 0
    next_state = env.reset()
    with tqdm(total=num_epochs, leave=False, unit='Ep', postfix="") as pbar:
        while t_ep <= num_epochs:
            for t in batch_counter:
                states[t] = next_state.copy()
                actions[t], values[t] = pi.select_action(states[t])

                next_state, rewards[t], dones[t], _ = env.step(actions[t])
                episode_rewards[t_ep] += rewards[t]

                timestep += 1
                rewards[t] = adjust_reward(rewards[t], timestep, dones[t], env._max_episode_steps, adjust_factor, adjust)

                pi.sample_transition((states[t], actions[t], rewards[t], next_state, dones[t]))

                if dones[t]:
                    timestep = 0
                    t_ep += 1
                    pbar.set_postfix({'Mean recent R':
                                          f"{np.mean(episode_rewards[np.clip(t_ep-50, a_min=0, a_max=None):t_ep]):02f}"})
                    pbar.update(1)
                    next_state = env.reset()
            losses = pi.update_policy(states, actions, rewards, values, dones, next_state)
    pbar.close()
    return episode_rewards[:num_epochs]


if __name__ == '__main__':
    import gym
    from tensorflow.keras.utils import plot_model
    from scipy.signal import savgol_filter

    env = gym.make('CartPole-v1')
    env._max_episode_steps = 200

    rewards = train_actor_critic(env, num_epochs=500,
                                 use_BS=True,
                                 use_BLS=True,
                                 use_ER=False,
                                 use_TES=False,
                                 use_AN=False,
                                 hidden_layers_actor=(32, 8),
                                 hidden_layers_critic=(32, 8),
                                 hidden_act_actor='tanh',
                                 kernel_init_actor="glorot_uniform", actor_output_activation=None,
                                 hidden_act_critic='relu',
                                 kernel_init_critic="glorot_uniform", critic_output_activation=None,
                                 )

    plt.plot(savgol_filter(rewards, window_length=51, polyorder=1), linestyle="solid")
    plt.plot(savgol_filter(rewards, window_length=3, polyorder=1), alpha=0.5, linestyle="dashed", linewidth=0.75)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.show()
