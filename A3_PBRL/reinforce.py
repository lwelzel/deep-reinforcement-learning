#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Reinforce Class
#
#######################

import numpy as np
import gym
from base_agent import BaseAgent

class ReinforceAgent(BaseAgent):
    def __init__(self, state_space, action_space, **kwargs):
        super().__init__(state_space, action_space, **kwargs)
        
    def update(self, trace_array, episode_len):
        print(f"Average Reward: {np.mean(episode_len)}")

   



def custom_train_reinforce():
    """Function to train a REINFORCE agent on the cartpole-v1 environment for testing"""

    ##### PARAMETERS #####
    num_traces = 5
    ######################

    env = gym.make('CartPole-v1')
    pi  = ReinforceAgent(env.observation_space, env.action_space)
    
    




def main():
    custom_train_reinforce()


if __name__ == '__main__':
    main()
