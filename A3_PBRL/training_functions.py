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
<<<<<<< HEAD



=======
from reinforce import ReinforceAgent
from actor_critic import ActorCriticAgent
from evolutionary import EvolutionaryAgent


>>>>>>> 24cd2c39f34e1ae7169a03bca6befb634374356c
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


<<<<<<< HEAD
def train(method, num_epochs=200, num_traces=5, num_agents=25, 
          max_reward = 200,
          verbose=False, render=False,
          save_rewards=False, save_freq=10, 
          **kwargs):
=======
def train(method, train_length=100, n_traces=5, verbose=False, render=False,
          save_rewards=False, save_freq=10, num_agents=25, **kwargs):
>>>>>>> 24cd2c39f34e1ae7169a03bca6befb634374356c
    """General Training function"""

    t_start = time.time()

    env = gym.make('CartPole-v1')
    env._max_episode_steps = max_reward
    if method == 'reinforce':
    	from reinforce import ReinforceAgent
    	train_reinforce(env, **kwargs)
      
    elif method == 'actor-critic':
<<<<<<< HEAD
    	from actor_critic import ActorCriticAgent
        train_ac(env, **kwargs)
        
    elif method == 'evolutionary':
    	from evolutionary import EvolutionaryAgent
        train_evo(env, **kwargs)
        
    print(f"One full training iteration took {time.time() - t_start:.0f}" seconds)

=======
        pi = ActorCriticAgent(env.observation_space, env.action_space, **kwargs)
    elif method == 'evolutionary':
        pi = EvolutionaryAgent(env.observation_space, env.action_space, num_agents, **kwargs)
>>>>>>> 24cd2c39f34e1ae7169a03bca6befb634374356c


def train_reinforce(env, num_epochs=200, num_traces=5,
                    verbose=False, render=False,
                    save_rewards=False, save_freq=5,
                    **kwargs):
                    
    pi = ReinforceAgent(env.observation_space, env.action_space, name='reinforce', **kwargs)
    average_trace_reward = np.zeros(train_length)
<<<<<<< HEAD
    
    for epoch in range(num_epochs):
        trace_array, episode_len = sample_traces(env, pi, n_traces, render)
        _ = pi.update_policy(trace_array, episode_len)
        average_trace_reward[epoch] = np.mean(episode_len)
        
=======
    for epoch in range(train_length):
        t_start = time.time()

        if method == 'evolutionary':
            agent_rewards = np.zeros(num_agents)
            for i in range(num_agents):
                pi.set_agent(i)
                _, episode_len = sample_traces(env, pi, n_traces, render)
                agent_rewards[i] = np.mean(episode_len)
                pi.collect_return(i, np.mean(episode_len))
            loss = pi.update_policy()

            average_trace_reward[epoch] = np.mean(agent_rewards)
        else:
            trace_array, episode_len = sample_traces(env, pi, n_traces, render)

            loss = pi.update_policy(trace_array, episode_len)
            average_trace_reward[epoch] = np.mean(episode_len)

>>>>>>> 24cd2c39f34e1ae7169a03bca6befb634374356c
        if (epoch % save_freq) == 0 and save_rewards:
            pi.save(average_trace_reward[:epoch])
            
        if verbose:
<<<<<<< HEAD
            print(f"Epoch {epoch} |  Elapsed Time: {time.time() - t_start:.2f}s | Mean Reward: {average_trace_reward[epoch]:.1f}"
            
=======
            if pi.exp_policy == 'egreedy':
                exp_factor = pi.epsilon
            elif pi.exp_policy == 'softmax':
                exp_factor = pi.temp
            print(
                f"Completed Iteration {epoch} | Elapsed Time: {time.time() - t_start:.2f}s | Mean Reward: {average_trace_reward[epoch]:.1f} | Exploration Factor: {exp_factor:.2f} ({pi.exp_policy}) | Loss: {loss:.3f}")

>>>>>>> 24cd2c39f34e1ae7169a03bca6befb634374356c
        pi.anneal_policy_parameter(epoch, train_length)
    
    if save_rewards:
        pi.save(average_trace_reward)
        

def train_evo(env, num_epochs=20, num_traces=5, num_agents=50,
              verbose=False, render=False,
              save_rewards=False, save_freq=2,
              **kwargs):
                    
    pi = EvolutionaryAgent(env.observation_space, env.action_space, num_agents, name='evolutionary', **kwargs)
    average_trace_reward = np.zeros(train_length)
    
    for epoch in range(num_epochs):
            agent_rewards = np.zeros(num_agents)
            for i in range(num_agents):
                pi.set_agent(i)
                _, episode_len = sample_traces(env, pi, n_traces, render)
                agent_rewards[i] = np.mean(episode_len)
                pi.collect_return(i, np.mean(episode_len))
            _ = pi.update_policy()

            average_trace_reward[epoch] = np.mean(agent_rewards)   
        
        if (epoch % save_freq) == 0 and save_rewards:
            pi.save(average_trace_reward[:epoch])
            
        if verbose:
            print(f"Epoch {epoch} |  Elapsed Time: {time.time() - t_start:.2f}s | Mean Reward: {average_trace_reward[epoch]:.1f}"
            
        pi.anneal_policy_parameter(epoch, train_length)
    
    if save_rewards:
        pi.save(average_trace_reward)
        
    

def main():
<<<<<<< HEAD
    train('evolutionary', train_length=200, n_traces=5, verbose=True, render=True, save_rewards=True, save_freq=10,
          num_agents=40, fit_method='individual', anneal_method='exponential', epsilon=0.7, decay=0.9)
=======
    train('evolutionary', train_length=200, n_traces=5, verbose=True, render=True, save_rewards=True, save_freq=100,
          num_agents=40, fit_method='individual', exp_policy=None, epsilon=0.2)
>>>>>>> 24cd2c39f34e1ae7169a03bca6befb634374356c


if __name__ == '__main__':
    main()
