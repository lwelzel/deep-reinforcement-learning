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

            transition = np.zeros((4,4))
            for j, obs in enumerate([s, a, r, s_next]):
                transition[j, :] = obs
            trace_array[i, 4*t:4*(t+1), :] = transition

            t += 1
            if verbose: env.render()
            if t == pi.max_reward: done = True

        episode_len[i] = t

    return trace_array, episode_len


def train(method, verbose=False):
    """General training function"""
    env = gym.make('CartPole-v1')
    if method == 'reinforce':
        pi = ReinforceAgent(env.observation_space, env.action_space)
    elif method == 'actor-critic':
        pi = ActorCriticAgent(env.observation_space, env.action_space)

    # learning process (NOT COMPLETED)
    converged = False
    while not converged:
        trace_array, episode_len = sample_traces(env, pi, 5, verbose)
        weight_grad = pi.update(trace_array, episode_len)
        converged = pi.update_weights(weight_grad)


def main():
    train('reinforce', verbose=True)


if __name__ == '__main__':
    main()
