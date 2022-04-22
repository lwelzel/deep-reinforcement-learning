#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Reinforce Class
#
#######################

import numpy as np
import gym

class Reinforce_Agent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__()
        
        
    def update():
        pass;

   
def get_traces(env, pi, num_traces):
    pass;


def custom_train_reinforce():
    """Function to train a REINFORCE agent on the cartpole-v1 environment for testing"""

    ##### PARAMETERS #####
    num_traces = 5
    ######################

    env = gym.make('CartPole-v1')
    pi  = Reinforce_Agent()
    
    




def main():
    custom_train_reinforce()


if __name__ == '__main__':
    main()
