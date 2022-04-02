#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from many_dqn_wrapper import run_parallel_dqns
from dqn import DeepQAgent, learn_dqn
from helper import LearningCurvePlot, smooth

def average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                             use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq,anneal_method):

    reward_results = np.empty([n_repetitions,num_iterations]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        rewards = run_parallel_dqns(learning_rate,policy,epsilon,temp,gamma,hidden_layers,use_er,use_tn,num_iterations,depth,learn_freq,\
                  target_update_freq,sample_batch_size=128,anneal_method=None,render=False)
        
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
    anneal_method = None
    
    hidden_layers = [32,32]
    depth = 2500
    batch_size = 128
    num_iterations = 1000
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
    do_layers = False
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
        eg_exp_ar = [[0.01,None],[0.1,None],[0.25,None],[1.,'linear']]
        sm_exp_ar = [0.01,0.1,1.]
        for epsilon,anneal_method in exp_ar:
            lc = average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                        use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq,anneal_method)
            if anneal_method == None:
                label = r"$\epsilon$ = "+str(epsilon)
            else:
                label = r"$\epsilon$-greedy Linear Annealing"
            Plot.add_curve(lc,label=label)

        policy = 'softmax'
        for temp in sm_exp_ar:
            lc = average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
            use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq,anneal_method)
            Plot.add_curve(lc,label=r'$\tau$ = '+str(temp)      

        Plot.save('exploration.png')
        
        policy = 'egreedy'
        epsilon, temp, anneal_method = 0.1, 1., None

    if do_layers:
        layers_ar = [[64],[48,24],[32,32]]
        for hidden_layers in layers_ar:
            lc = average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
            use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq,anneal_method)
            Plot.add_curve(lc,label=hidden_layers)
        Plot.save('archictecture.png')

        hidden_layers = [32,32]

if __name__ == '__main__':
    main()
