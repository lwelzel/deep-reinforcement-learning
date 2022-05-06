#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Training functions taking policy search class as input
#
#######################

import numpy as np
import gym
import time

from reinforce import ReinforceAgent
from actor_critic import ActorCriticAgent
from evolutionary import EvolutionaryAgent



def sample_traces(env, pi, n_traces, render=False):
    # create array that could potentially contain all complete traces
    trace_array = np.zeros((n_traces, 4 * pi.max_reward, 4))
    episode_len = np.zeros(n_traces, dtype=int)  # track how many entries for each trace

    # simulate iteratively
    for i in range(n_traces):
        s = env.reset()
        done = False
        t = 0
        while not done:
            a, _ = pi.select_action(s)
            s_next, r, done, _ = env.step(a)
            if t == pi.max_reward:
                done = True

            transition = np.zeros((4, 4))
            for j, obs in enumerate([s, a, r, s_next]):
                transition[j, :] = obs
            trace_array[i, 4 * t:4 * (t + 1), :] = transition

            s = s_next
            t += 1
            if render:
                env.render()

        episode_len[i] = t

    return trace_array, episode_len


def train(method, num_epochs=200, num_traces=5, num_agents=25, 
          max_reward = 200,
          verbose=False, render=False,
          save_rewards=False, save_freq=10, 
          **kwargs):
    """General Training function"""

    t_start = time.time()

    env = gym.make('CartPole-v1')
    env._max_episode_steps = max_reward
    if method == 'reinforce':
        train_reinforce(env, **kwargs)
    elif method == 'actor-critic':
        train_ac(env, **kwargs)
    elif method == 'evolutionary':
        train_evo(env, **kwargs)
        
    print(f"One full training iteration took {time.time() - t_start:.0f} seconds")



def train_reinforce(env, num_epochs=200, num_traces=5,
                    verbose=False, render=False,
                    save_rewards=False, save_freq=5,
                    **kwargs):
                    
    pi = ReinforceAgent(env.observation_space, env.action_space, **kwargs)
    average_trace_reward = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        trace_array, episode_len = sample_traces(env, pi, num_traces, render)
        _ = pi.update_policy(trace_array, episode_len)
        average_trace_reward[epoch] = np.mean(episode_len)
        
        if (epoch % save_freq) == 0 and save_rewards:
            pi.save(average_trace_reward[:epoch])
            
        if verbose:
            print(f"Epoch {epoch} |  Elapsed Time: {time.time() - t_start:.2f}s | Mean Reward: {average_trace_reward[epoch]:.1f}")
            
        pi.anneal_policy_parameter(epoch, num_epochs)
    
    if save_rewards:
        pi.save(average_trace_reward)
        

def train_evo(env, num_epochs=20, num_traces=5, num_agents=50,
              verbose=False, render=False,
              save_rewards=False, save_freq=2,
              **kwargs):
                    
    pi = EvolutionaryAgent(env.observation_space, env.action_space, num_agents, **kwargs)
    average_trace_reward = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        agent_rewards = np.zeros(num_agents)
        for i in range(num_agents):
            pi.set_agent(i)
            _, episode_len = sample_traces(env, pi, num_traces, render)
            agent_rewards[i] = np.mean(episode_len)
            pi.collect_return(i, np.mean(episode_len))
        _ = pi.update_policy()

        average_trace_reward[epoch] = np.mean(agent_rewards)   
        
        if (epoch % save_freq) == 0 and save_rewards:
            pi.save(average_trace_reward[:epoch])
            
        if verbose:
            print(f"Epoch {epoch} |  Elapsed Time: {time.time() - t_start:.2f}s | Mean Reward: {average_trace_reward[epoch]:.1f}")
            
        pi.anneal_policy_parameter(epoch, num_epochs)
    
    if save_rewards:
        pi.save(average_trace_reward)
        
    

def main():
    train('evolutionary', num_epochs=200, num_traces=5, verbose=True, render=True, save_rewards=True, save_freq=10,
          num_agents=40, fit_method='individual', anneal_method='exponential', epsilon=0.7, decay=0.9)


if __name__ == '__main__':
    main()
