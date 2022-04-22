#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Training functions taking policy search class as input
#
#######################

import numpy as np
from reinforce import ReinforceAgent
from actor_critic import ActorCriticAgent


def sample_traces(env, pi, n_traces, max_episode_length=200):
    trace_array = np.zeros((n_traces, 4*max_episode_length)) #create array that could potentially contain all complete traces
    episode_len = np.zeros(n_traces) #track how many entries for each trace

    #simulate iteratively
    for i in range(n_traces):
        s = env.reset()
        done = False
        t = 0
        while not done:
            a = pi.select_action(s)
            s_next, r, done, _ = env.step(a)
            trace_array[i, 4*t:4*(t+1)] = s, a, r, s_next
            t += 1
            if t == max_episode_length: done = True
        episode_len[i] = t

    return trace_array, episode_len
    

def train(method):
    """General training function"""
    env = gym.make('CartPole-v1')
    if method == 'reinforce':
        pi = ReinforceAgent(env.observation_space,env.action_space)
    elif method == 'actor-critic':
        pi = ActorCriticAgent(env.observation_space,env.action_space)

    #learning process (NOT COMPLETED)
    trace_array, episode_len = sample_traces(env, pi, 5)
    pi.update(trace_array, episode_len)
    


def main():
    pass;


if __name__ == '__main__':
    main()
