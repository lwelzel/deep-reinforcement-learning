#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from many_dqn_wrapper import run_parallel_dqns
from dqn import DeepQAgent, learn_dqn
from helper import LearningCurvePlot, smooth

def average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                             use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq):

    reward_results = np.empty([n_repetitions,num_iterations]) # Result array
    now = time.time()
    max_epoch_env_steps = 250    

    for rep in range(n_repetitions): # Loop over repetitions
        rewards = run_parallel_dqns(num_iterations,max_epoch_env_steps,target_update_freq,
                      policy,
                      learning_rate, gamma,
                      epsilon, temp,
                      hidden_layers, hidden_act='relu', kernel_init='HeUniform',
                      loss_func='mean_squared_error',
                      use_tn=use_tn, use_er=use_er,
                      buffer_type=None, buffer_depth=depth, sample_batch_size=100)
        
        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,window=51) # additional smoothing
    return learning_curve

def main():
    
    ###PARAMETERS###############
    learning_rate=0.01
    gamma=0.8
    
    epsilon = .1
    temp = 1.
    policy = 'egreedy'#'softmax'#
       
    hidden_layers = [32,32]
    depth = 2500
    batch_size = 128
    num_iterations = 500
    target_update_freq = 25  # iterations
    learn_freq=4
    max_training_batch = int(1e6)  # storage arrays
    # training_data_shape = (max_training_batch, 1)

    e_anneal = False
    use_er = True
    use_tn = True

    render = False
    plot = True
    title = r"Softmax $\tau$=1, +TN -ER"

    do_ablation = False
    do_exploration = True
    do_layers = True
    ###########################
    n_repetitions = 5    


    if do_ablation:
        Plot = LearningCurvePlot(title = "Deep Q-Learning, Ablation Study")
        
        
        learning_curve0 = average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                                use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)
        Plot.add_curve(learning_curve0,label=r'DQN')
        
        use_er = True
        use_tn = False
        learning_curve1 = average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                                use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)
        Plot.add_curve(learning_curve1,label=r'DQN without TN')
        
        use_er = False
        use_tn = True
        learning_curve2 = average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                                use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)
        Plot.add_curve(learning_curve2,label=r'DQN without ER')
        
        use_er = False
        use_tn = False
        learning_curve3 = average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                                use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)
        Plot.add_curve(learning_curve3,label=r'DQN without TN and ER')
        
        Plot.save('Ablation.png')
        
        # Restore previous values
        use_er = True
        use_tn = True


    if do_exploration:
        Plot = LearningCurvePlot(title = "Deep Q-Learning, Exploration Study")
        eg_exp_ar = [0.01,0.1,0.25,1.]

        for epsilon in eg_exp_ar:
            lc = average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                        use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)
            Plot.add_curve(lc,label=r"$\epsilon$ = "+str(epsilon))

        policy = 'softmax'
        lc = average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
        use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)
        Plot.add_curve(lc,label=r'$\tau$ = '+str(temp))

        Plot.save('exploration.png')
        
        policy = 'egreedy'
        epsilon, temp = 0.1, 1.

    if do_layers:
        Plot = LearningCurvePlot(title = "Deep Q-Learning, Hidden Layer Study")
        layers_ar = [[24,12],[48,24],[32,32]]
        for hidden_layers in layers_ar:
            lc = average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
            use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)
            Plot.add_curve(lc,label=hidden_layers)
        Plot.save('archictecture.png')

        hidden_layers = [32,32]

if __name__ == '__main__':
    main()
