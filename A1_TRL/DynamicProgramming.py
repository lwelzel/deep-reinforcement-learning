#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax


class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s):
        ''' Returns the greedy best action in state s '''
        a = argmax(self.Q_sa[s])
        return a

    def update(self, s, a, p_sas, r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        self.Q_sa[s, a] = np.sum(
            p_sas
            * (
                    r_sas
                    + self.gamma * np.max(self.Q_sa, axis=1)
            )
        )
        pass

def Q_value_iteration(env, gamma=1.0, threshold=0.001, show=False, return_iters=False, iter_stop=100):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''

    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    max_error = np.inf
    i = 0
    while max_error > threshold:
        max_error = 0
        for s in np.arange(QIagent.n_states):
            for a in np.arange(QIagent.n_actions):
                x = QIagent.Q_sa[s, a]
                p_sas, r_sas = env.model(s, a)
                QIagent.update(s, a, p_sas, r_sas)
                max_error = np.max((max_error, np.max(np.abs(x - QIagent.Q_sa[s, a]))))
        i += 1

        if show:
            # Plot current Q-value estimates & print max error
            env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
            print(f"Q-value iteration, iteration {i:03.0f}, max error {max_error:.20e}")
        if i > iter_stop:
            break

    if return_iters:
        return QIagent, i
    return QIagent


def experiment():
    gamma =  1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env, gamma, threshold, show=True)

    reward = []

    # View optimal policy
    done = False
    s = env.reset()
    i = 0
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        reward.append(r)
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.05)
        s = s_next
        i += 1
        if i > 25:
            break

    print()

    # this is higher since its ignoring that environment is stochastic
    mean_steps = env.goal_reward - np.max(QIagent.Q_sa[env._location_to_state(np.array([0, 3]))]) + 1
    print(f"Expected mean reward per timestep "
          f"under optimal policy: {np.mean(np.max(QIagent.Q_sa, axis=1) / mean_steps):.3f}")

    mean_reward_per_timestep = np.mean(reward)
    print(f"Mean reward per timestep, actual (this run):            {mean_reward_per_timestep:.3f}")

    env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=5)

if __name__ == '__main__':
    experiment()
