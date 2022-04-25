#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Training functions taking policy search class as input
#
#######################

import numpy as np
import gym
from reinforce import ReinforceAgent
from actor_critic import ActorCriticAgent


def sample_traces(env, pi, n_traces, verbose=False):
    # create array that could potentially contain all complete traces
    trace_array = np.zeros((n_traces, 4 * pi.max_reward, 4))
    episode_len = np.zeros(n_traces)  # track how many entries for each trace

    # simulate iteratively
    for i in range(n_traces):
        s = env.reset()
        done = False
        t = 0
        while not done:
            a = pi.select_action(s)
            s_next, r, done, _ = env.step(a)

            transition = np.zeros((4, 4))
            for j, obs in enumerate([s, a, r, s_next]):
                transition[j, :] = obs
            trace_array[i, 4 * t:4 * (t + 1), :] = transition

            t += 1
            if verbose: env.render()
            if t == pi.max_reward: done = True

        episode_len[i] = t

    return trace_array, episode_len


def train(method, train_length=100, n_traces=5, verbose=False,
          save_rewards=False, save_freq=10, **kwargs):
    """Training function"""

    env = gym.make('CartPole-v1')
    if method == 'reinforce':
        pi = ReinforceAgent(env.observation_space, env.action_space, **kwargs)
    elif method == 'actor-critic':
        pi = ActorCriticAgent(env.observation_space, env.action_space, **kwargs)

    average_trace_reward = np.zeros(train_length)
    for i in range(train_length):
        trace_array, episode_len = sample_traces(env, pi, n_traces, verbose)
        pi.update_policy(trace_array, episode_len)
        average_trace_reward[i] = np.mean(episode_len)

        if (i % save_freq) == 0 and save_rewards:
            pi.save(average_trace_reward[:i])

    if save_rewards:
        pi.save(average_trace_reward)


def main():
    train('reinforce', train_length=100, verbose=True)


if __name__ == '__main__':
    main()
