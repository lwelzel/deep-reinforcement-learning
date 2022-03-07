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
from Helper import softmax, argmax, plot_q_value_func, AnnealScheduler


class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
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

    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        done_tracker = np.full(states.size, fill_value=False)
        done_tracker[-1] = done

        for time in np.arange(actions.size):
            m = np.min((self.n, actions.size - time))

            G = np.sum(np.power(self.gamma, np.arange(m)) * rewards[time:time + m])\
                + ~ done_tracker[time + m] * np.power(self.gamma, m) * np.max(self.Q_sa[states[time + m]])

            self.Q_sa[states[time], actions[time]] = self.Q_sa[states[time], actions[time]]\
                                                     + self.learning_rate * (G - self.Q_sa[states[time], actions[time]])
        pass


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
             policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an nstep rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)

    rewards = np.full(n_timesteps + max_episode_length + 1, fill_value=np.nan)
    states_tracker = np.zeros(max_episode_length + 1, dtype=int)
    states_tracker[0] = env.reset()
    actions_tracker = np.zeros(max_episode_length, dtype=int)

    time_total = 0
    while time_total < n_timesteps:
        s_next = env.reset()
        for time in np.arange(max_episode_length):
            s = s_next
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
            s_next, r, done = env.step(a)
            rewards[time_total + time] = r
            states_tracker[time + 1] = s_next
            actions_tracker[time] = a
            if (done is True):
                break
        pi.update(states_tracker[:time + 2], actions_tracker[:time + 1], rewards[time_total: time_total + time + 1], done)
        time_total += time + 1

    # if plot:
    #    # env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
    #    plot_q_value_func(Q_sa=pi.Q_sa, env=env, title=None)

    return rewards[:n_timesteps]  # cut of overstepping for safety

def n_step_Q_annealing(n_timesteps, max_episode_length, learning_rate, gamma,
             policy='egreedy', epsilon=None, temp=None, plot=True, n=5,
                       anneal_schedule="linear_anneal",
                       buffer=25, start=1, final=0.0, Q_thresh=0.05, r_thresh=0.05, percentage=0.5):
    ''' runs a single repetition of an nstep rl agent with annealing
        Return: rewards, a vector with the observed rewards at each timestep '''
    # "q_error_anneal"
    # "r_diff_anneal"
    # "linear_anneal"
    # "logistic_anneal"



    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)

    schedule = AnnealScheduler(buffer,
                               start=start,
                               final=final,
                               Q_tresh=Q_thresh,
                               timesteps=n_timesteps,
                               r_thresh=r_thresh,
                               percentage=percentage)

    rewards = np.full(n_timesteps + max_episode_length + 1, fill_value=np.nan)
    states_tracker = np.zeros(max_episode_length + 1, dtype=int)
    states_tracker[0] = env.reset()
    actions_tracker = np.zeros(max_episode_length, dtype=int)

    scheduler = schedule.function[anneal_schedule]
    para = start

    time_total = 0
    while time_total < n_timesteps:
        s_next = env.reset()
        for time in np.arange(max_episode_length):
            s = s_next
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
            s_next, r, done = env.step(a)
            rewards[time_total + time] = r
            states_tracker[time + 1] = s_next
            actions_tracker[time] = a
            if (done is True):
                break

        pi.n = int(scheduler(new_Q_sa=pi.Q_sa, new_r=np.mean(rewards[time_total: time_total + time + 1]),
                             t=time_total + time, k=1))
        pi.update(states_tracker[:time + 2], actions_tracker[:time + 1], rewards[time_total: time_total + time + 1], done)
        time_total += time + 1

    # if plot:
    #    # env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
    #    plot_q_value_func(Q_sa=pi.Q_sa, env=env, title=None)

    return rewards[:n_timesteps]  # cut of overstepping for safety


def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
                       policy, epsilon, temp, plot, n=n)
    print(rewards.shape)
    print("Obtained rewards: {}".format(np.mean(rewards)))


if __name__ == '__main__':
    test()
