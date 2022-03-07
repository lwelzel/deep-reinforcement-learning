#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from tqdm import tqdm
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax, AnnealScheduler
import warnings


class QLearningAgent:

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

        try:
            return self.rng.choice(self.n_actions, None, p=self.arr_prob)
        except ValueError:
            # some of the annealers are a *little* unstable sometimes, fall back on greedy
            warnings.warn("Invalid value due to annealing scheduler. Falling back on greedy policy for this step.")
            return argmax(self.Q_sa[s])

    def update(self, s, a, r, s_next, done):
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate \
                          * (r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a])


def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy',
               epsilon=None, temp=None, plot=True, ret_Q_sa=False,
               max_episode_length=100, ret_full_rewards=False):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = np.full(n_timesteps, fill_value=np.nan)

    t_episode = -1
    s_next = env.reset()
    for time in tqdm(np.arange(n_timesteps), leave=False):
        s = s_next
        a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
        s_next, r, done = env.step(a)
        rewards[time] = r
        pi.update(s, a, r, s_next, done)
        if (done is True) or ((time - t_episode) % max_episode_length == 0):
            t_episode = time
            s_next = env.reset()
            continue

    if plot:
        __ = env.reset()
        env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
                   step_pause=0.5)  # Plot the Q-value estimates during Q-learning execution

    if ret_Q_sa:
        return rewards, pi.Q_sa

    return rewards


def q_learning_anneal(n_timesteps, learning_rate, gamma, policy='egreedy',
                      epsilon=None, temp=None, plot=True, ret_Q_sa=False,
                      max_episode_length=100, ret_full_rewards=False,
                      anneal_schedule="linear_anneal",
                      buffer=25, start=0.3, final=0.0, Q_thresh=0.05, r_thresh=0.05, percentage=0.5):
    # "q_error_anneal"
    # "r_diff_anneal"
    # "linear_anneal"
    # "logistic_anneal"

    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    schedule = AnnealScheduler(buffer,
                               start=start,
                               final=final,
                               Q_tresh=Q_thresh,
                               timesteps=n_timesteps,
                               r_thresh=r_thresh,
                               percentage=percentage)

    scheduler = schedule.function[anneal_schedule]
    para = start

    rewards = np.full(n_timesteps + max_episode_length + 1, fill_value=np.nan)
    s_next = env.reset()

    time_total = 0
    while time_total < n_timesteps:
        for time in np.arange(max_episode_length):
            s = s_next
            a = pi.select_action(s, policy=policy, epsilon=para, temp=para)
            s_next, r, done = env.step(a)
            rewards[time_total + time] = r
            pi.update(s, a, r, s_next, done)
            if (done is True):
                s_next = env.reset()
                break

        para = scheduler(new_Q_sa=pi.Q_sa, new_r=np.mean(rewards[time_total: time_total + time + 1]), t=time_total + time, k=1)
        time_total += time + 1

    return rewards[:n_timesteps]


def test():
    n_timesteps = 20000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = False

    # rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    # print("Obtained rewards: {}".format(rewards))

    rewards = q_learning_anneal(n_timesteps, learning_rate, gamma, policy)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    from Helper import smooth
    ax.plot(smooth(rewards, 1001))
    plt.show()


if __name__ == '__main__':
    test()
