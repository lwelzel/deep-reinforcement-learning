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


class ActorCriticAgent(object):
    def __init__(self, state_space, action_space,
                 max_reward=500,
                 exp_policy='egreedy', epsilon=1., temperature=1.,
                 anneal_method='exponential',
                 decay=0.999, epsilon_min=0.01, temp_min=0.1,
                 learning_rate=5e-3, discount=0.95,
                 hidden_layers_actor=(32, 16), hidden_act_actor='relu',
                 kernel_init_actor="glorot_uniform", actor_output_activation=None,
                 hidden_layers_critic=(32, 16), hidden_act_critic='relu',
                 kernel_init_critic="glorot_uniform", critic_output_activation=None,
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
        # regularization coefficients
        self.value_rc = 0.5
        self.entropy_rc = 1e-4

        # TF functions
        # get a weighted, sparse, cross entropy function to compute cross entropy for loss function
        # We have "softmax" as the output of our action_probability (actor) network branch.
        # This means that the output probabilities of out actor are already normalized and not logits (what a shitty name btw)
        self.weighted_cce_fn = SparseCategoricalCrossentropy(from_logits=True,
                                                             reduction="auto",
                                                             name="weighted_sparse_CCE_softmax_input")
        self.cce_fn = CategoricalCrossentropy(from_logits=False,
                                              reduction="auto",
                                              name="CCE_softmax_input")

        # IMPORTANT: If the model 'actor_prop_rescale_policy' is not a function that normalizes the output probabilities
        # you need to change the 'from_logits' option in 'SparseCategoricalCrossentropy' to 'True'
        self.model = ActorCriticModel(self.action_state_size, self.state_space_size,
                                      hidden_layers_actor=hidden_layers_actor, hidden_act_actor=hidden_act_actor,
                                      kernel_init_actor=kernel_init_actor,
                                      hidden_layers_critic=hidden_layers_critic, hidden_act_critic=hidden_act_critic,
                                      kernel_init_critic=kernel_init_critic,
                                      actor_output_activation=actor_output_activation,
                                      critic_output_activation=critic_output_activation)  # softmax

        self.model.compile(optimizer=RMSprop(lr=learning_rate),
                           loss=[self.policy_loss, self.value_loss], )

    def select_action(self, s, num_samples=1):
        action_probability, value = self.model(s.reshape(-1, 4))
        action = tf.squeeze(tf.random.categorical(action_probability, num_samples=num_samples))
        return action, value

    @staticmethod
    @njit(parallel=False, nogil=True, fastmath=True)
    def get_discounted_rewards(rewards, dones, gamma, predicted_value=1.):
        """
        Compute discounted rewards using rewards over trace
        """
        # TODO: https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning

        returns = np.zeros(len(rewards) + 1)
        returns[-1] = predicted_value
        for t in np.flip(np.arange(len(rewards))):
            returns[t] = rewards[t] + gamma * np.roll(returns, -1)[t] * (1 - dones[t])

        return returns[:-1]

    def get_advantage(self, target_values, rewards, dones, next_value):
        returns = self.get_discounted_rewards(rewards, dones, self.gamma, next_value)
        return returns - target_values, returns

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
        # TODO: figure out if I can rescale the output from NN
        action_probability = tf.nn.softmax(action_probability)

        entropy_loss = self.cce_fn(action_probability,
                                   action_probability)

        return policy_loss - entropy_loss * self.entropy_rc

    def update_policy(self, trace_array, actions, rewards, values, dones, next_state):
        __, next_value = self.model(np.reshape(next_state, (-1, self.state_space_size)))
        advantages, returns = self.get_advantage(target_values=values,
                                                 rewards=rewards,
                                                 dones=dones,
                                                 next_value=float(next_value))

        actions_advantage = np.stack((actions, advantages), axis=-1)

        losses = self.model.train_on_batch(trace_array,
                                           [actions_advantage,
                                            returns])
        return losses


def train_actor_critic(env, num_epochs, batch_size=64, **kwargs):
    pi = ActorCriticAgent(env.observation_space, env.action_space, **kwargs)
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
    next_state = env.reset()
    with tqdm(total=num_epochs, leave=False, unit='Ep', postfix="") as pbar:
        while t_ep <= num_epochs:
            for t in batch_counter:
                states[t] = next_state.copy()
                actions[t], values[t] = pi.select_action(states[t])

                next_state, rewards[t], dones[t], _ = env.step(actions[t])
                episode_rewards[t_ep] += rewards[t]

                if dones[t]:
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
                                 hidden_layers_actor=(256, 64, 16),
                                 hidden_layers_critic=(256, 64, 16))

    plt.plot(savgol_filter(rewards, window_length=51, polyorder=1), linestyle="solid")
    plt.plot(savgol_filter(rewards, window_length=3, polyorder=1), alpha=0.5, linestyle="dashed", linewidth=0.75)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.show()
