#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Catch all training- and sampling function
#  Individual training loops for REINFORCE and CEM
#
#######################

import numpy as np
import gym
import time

from reinforce import ReinforceAgent
from actor_critic import train_actor_critic
from evolutionary import EvolutionaryAgent
from tqdm import tqdm


def sample_traces(env, pi, n_traces, render=False):
    # create array that could potentially contain all complete traces
    trace_array = np.zeros((n_traces, 4 * pi.max_reward, 4))
    episode_len = np.zeros(n_traces, dtype=int)  # track how many entries for each trace = reward

    # simulate and fill trace_array iteratively
    for i in range(n_traces):
        s = env.reset()
        done = False
        t = 0
        while not done:
            a, _ = pi.select_action(s)
            s_next, r, done, _ = env.step(a)
            if t == pi.max_reward:
                done = True

            # s, s_next are lists with len 4, a, r are integers. Need some padding to fill out 
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
          max_reward=200,
          verbose=False, render=False,
          save_rewards=False, save_freq=10, 
          **kwargs):
    """Main training function creating the environment, and pointing through to correct """

    t_start = time.time()

    env = gym.make('CartPole-v1')
    env._max_episode_steps = max_reward
    if method == 'reinforce':
        train_reinforce(env, num_epochs, num_traces, verbose, render, save_rewards, save_freq, **kwargs)
    elif method == 'actor-critic':
        __ = train_actor_critic(env, max_reward, **kwargs)
    elif method == 'evolutionary':
        train_evo(env, num_epochs, num_traces, num_agents, verbose, render, save_rewards, save_freq, **kwargs)
        
    print(f"One full training iteration took {time.time() - t_start:.0f} seconds")


def train_reinforce(env, num_epochs=200, num_traces=5,
                    verbose=False, render=False,
                    save_rewards=False, save_freq=5,
                    **kwargs):
    """REINFORCE Training Algorithm"""                
    pi = ReinforceAgent(env.observation_space, env.action_space, **kwargs)
    average_trace_reward = np.zeros(num_epochs)

    with tqdm(total=num_epochs, leave=False, unit='Ep', postfix="") as pbar:
        for epoch in range(num_epochs):
            trace_array, episode_len = sample_traces(env, pi, num_traces, render)
            _ = pi.update_policy(trace_array, episode_len)
            average_trace_reward[epoch] = np.mean(episode_len)

            if (epoch % save_freq) == 0 and save_rewards:
                pi.save(average_trace_reward[:epoch])

            pi.anneal_policy_parameter(epoch, num_epochs)
            pbar.set_postfix({'Mean recent R':
                                  f"{np.mean(average_trace_reward[np.clip(epoch - 50, a_min=0, a_max=None):epoch]):02f}"})
            pbar.update(1)
            print(f"Epoch {epoch} : Mean Reward {average_trace_reward[epoch]}")
    if save_rewards:
        pi.save(average_trace_reward)
        

def train_evo(env, num_epochs=20, num_traces=5, num_agents=50,
              verbose=False, render=True,
              save_rewards=False, save_freq=2,
              **kwargs):
    """CEM-Evolutionary Training Algorithm"""                   
    pi = EvolutionaryAgent(env.observation_space, env.action_space, num_agents, **kwargs)
    average_trace_reward = np.zeros(num_epochs)

    with tqdm(total=num_epochs, leave=False, unit='Ep', postfix="") as pbar:
        for epoch in range(num_epochs):
            agent_rewards = np.zeros(num_agents)
            for i in range(num_agents):
                pi.set_agent(i) # set agent i as main network for sampling and collecting returns
                _, episode_len = sample_traces(env, pi, num_traces, render)
                agent_rewards[i] = np.mean(episode_len)
                pi.collect_return(i, np.mean(episode_len))
            _ = pi.update_policy() # create new generation

            average_trace_reward[epoch] = np.mean(agent_rewards) # mean over all num_agents*num_traces traces

            if (epoch % save_freq) == 0 and save_rewards:
                pi.save(average_trace_reward[:epoch])

            pi.anneal_policy_parameter(epoch, num_epochs)
            pbar.set_postfix({'Mean recent R':
                                  f"{np.mean(average_trace_reward[np.clip(epoch - 5, a_min=0, a_max=None):epoch]):02f}"})
            pbar.update(1)
    
    if save_rewards:
        pi.save(average_trace_reward)
        
    

def main():
    train('reinforce', epsilon=0.2, anneal_method=None,  render=True, hidden_layers=[256, 256],
          save_rewards=True, save_freq=10,name='reinforce_softmax_hid_layers_256_256_lin_anneal_0p9')


if __name__ == '__main__':
    main()
