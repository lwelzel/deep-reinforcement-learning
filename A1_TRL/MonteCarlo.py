#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax, plot_q_value_func


class MonteCarloAgent:

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
            except KeyError:
                raise KeyError("No epsilon supplied in select_action().")
        elif policy == 'softmax':
            try:
                self.arr_prob[:] = softmax(self.Q_sa[s], temp)
            except KeyError:
                raise KeyError("No temperature supplied in select_action().")
        else:
            raise KeyError("No valid action supplied in select_action().")

        return self.rng.choice(self.n_actions, None, p=self.arr_prob)

    def update(self, states, actions, rewards,):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        '''

        for time in np.arange(actions.size):
            G = np.sum(np.power(self.gamma, np.arange(rewards.size - time)) * rewards[time:])

            self.Q_sa[states[time], actions[time]] = self.Q_sa[states[time], actions[time]] \
                                                     + self.learning_rate * (G - self.Q_sa[states[time], actions[time]])
        pass


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)

    rewards = np.full(n_timesteps + max_episode_length + 1, fill_value=np.nan)
    states_tracker = np.zeros(max_episode_length + 1, dtype=int)
    states_tracker[0] = env.reset()
    actions_tracker = np.zeros(max_episode_length, dtype=int)

    time_total = 0
    while time_total < n_timesteps:
        s_next = env.reset()
        # s_next = np.array([pi.rng.integers(0, 7), pi.rng.integers(0, 10)])
        for time in np.arange(max_episode_length):
            s = s_next
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
            s_next, r, done = env.step(a)
            rewards[time_total + time] = r
            states_tracker[time + 1] = s_next
            actions_tracker[time] = a
            if (done is True):
                break
        pi.update(states_tracker[:time + 2], actions_tracker[:time + 1], rewards[time_total: time_total + time + 1])
        time_total += time + 1

    if plot:
       # env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
       plot_q_value_func(Q_sa=pi.Q_sa, env=env, title=None)

    return rewards[:n_timesteps]  # cut of overstepping for safety


def test():
    n_timesteps = 1000000
    max_episode_length = 50
    gamma = 0.6
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                          policy, epsilon, temp, plot)




if __name__ == '__main__':
    test()
