#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from tqdm import tqdm
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax, plot_q_value_func


class SarsaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng()
        self.arr_prob = np.zeros(self.n_actions)

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        if policy == 'egreedy':  # switch over to try except, trying is cheap, if else is expensive
            try:
                # should precalculate this and then draw from matrix
                self.arr_prob[:] = epsilon / self.n_actions
                self.arr_prob[argmax(self.Q_sa[s])] = 1 - epsilon * ((self.n_actions - 1) / self.n_actions)
                # TO DO: Add own code
            except KeyError:
                raise KeyError("No epsilon supplied in select_action().")
        elif policy == 'softmax':
            try:
                self.arr_prob[:] = softmax(self.Q_sa[s], temp)
                # np.exp(self.Q_sa[s] / temp) / np.sum(np.exp(self.Q_sa[s] / temp))
            except KeyError:
                raise KeyError("No temperature supplied in select_action().")
        else:
            raise KeyError("No valid action supplied in select_action().")

        return self.rng.choice(self.n_actions, None, p=self.arr_prob)

    def update(self, s, a, r, s_next, a_next, done):
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate \
                          * (r + self.gamma * (not done) * self.Q_sa[s_next, a_next] - self.Q_sa[s, a])
        pass


def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = np.full(n_timesteps, fill_value=np.nan)

    max_episode_length = 100
    t_episode = -1

    s_next = env.reset()
    a_next = pi.select_action(s_next, policy=policy, epsilon=epsilon, temp=temp)
    for time in tqdm(np.arange(n_timesteps), leave=False):
        s, a = s_next, a_next
        s_next, r, done = env.step(a)
        a_next = pi.select_action(s_next, policy=policy, epsilon=epsilon, temp=temp)
        rewards[time] = r
        pi.update(s, a, r, s_next, a_next, done)
        if (done is True) or ((time - t_episode) % max_episode_length == 0):
            s_next = env.reset()
            a_next = pi.select_action(s_next, policy=policy, epsilon=epsilon, temp=temp)
            continue

    if plot:
        plot_q_value_func(Q_sa=pi.Q_sa, env=env, title=None)
        # env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=5) # Plot the Q-value estimates during SARSA execution


    return rewards


def test():
    n_timesteps = 250000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))


if __name__ == '__main__':
    test()
