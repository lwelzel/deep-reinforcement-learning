#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from many_dqn_wrapper import run_parallel_dqns
from dqn import DeepQAgent, learn_dqn

def average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                             use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq):
    
    now = time.time()
    max_epoch_env_steps = 200

    for rep in range(n_repetitions): # Loop over repetitions
        run_parallel_dqns(num_iterations,max_epoch_env_steps,target_update_freq,
                      policy,
                      learning_rate, gamma,
                      epsilon, temp,
                      hidden_layers, hidden_act='relu', kernel_init='HeUniform',
                      loss_func='mean_squared_error',
                      use_tn=use_tn, use_er=use_er,
                      buffer_type=None, buffer_depth=depth, sample_batch_size=100)
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    return

def main():
    
    ###PARAMETERS###############
    learning_rate=0.01
    gamma=0.9
    
    epsilon = 1. #starting epsilon, annealing
    temp = 1.
    policy = 'egreedy'#'softmax'#
       
    hidden_layers = [512,256,64]
    depth = 2000
    batch_size = 128
    num_iterations = 250
    target_update_freq = 4  # iterations
    learn_freq=4
    max_training_batch = int(1e6)  # storage arrays
    # training_data_shape = (max_training_batch, 1)

    e_anneal = False
    use_er = True
    use_tn = True

    render = False
    plot = True

    do_exploration = False
    do_layers = False
    do_learnrates = True
    do_gammas = False
    ###########################
    n_repetitions = 1   
    
    
    if do_learnrates:
        for learning_rate in [0.25,0.5]:
            average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                                use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)
        learning_rate=0.01
    
    if do_gammas:        
        for gamma in [0.1,0.3,0.6]:
            average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                                use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)
        gamma=0.9
    
    if do_exploration:
        policy = 'softmax'
        average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                                 use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)

        policy = 'egreedy'
        average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                                 use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)

    if do_layers:
        layers_ar = [[64,32],[128,64],[256,64],[256,128,64],[512,256,64]]
        for hidden_layers in layers_ar:
            average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                                     use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)
        hidden_layers = [512,256,64]

if __name__ == '__main__':
    main()
