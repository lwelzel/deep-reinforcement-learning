#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from dqn import DeepQAgent, learn_dqn
from helper import LearningCurvePlot, smooth

def average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                             use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq):

    reward_results = np.empty([n_repetitions,num_iterations]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        rewards = learn_dqn(learning_rate,policy,epsilon,temp,gamma,hidden_layers,use_er,use_tn,num_iterations,depth,learn_freq,\
                  target_update_freq,sample_batch_size=128,anneal_method="linear",render=False)
        
        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,window=51) # additional smoothing
    return learning_curve

def ablation():
    
    ###PARAMETERS###############
    learning_rate=0.01
    gamma=0.8
    
    epsilon = .1
    temp = 1.
    policy = 'egreedy'#'softmax'#
    
    hidden_layers = [12,6]
    depth = 250
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
    ###########################
    
    Plot = LearningCurvePlot(title = "Deep Q-Learning, Ablation Study")
    n_repetitions = 10
    
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

def learn_rate():
    
    ###PARAMETERS###############
    #learning_rate=0.01
    gamma=0.8
    
    epsilon = .1
    temp = 1.
    policy = 'egreedy'#'softmax'#
    
    hidden_layers = [12,6]
    depth = 250
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
    ###########################
    
    Plot = LearningCurvePlot(title = "Deep Q-Learning: Various Learning Rates Compared")
    n_repetitions = 10
    
    for learning_rate in [0.01,0.1,0.25,0.5]:
        learning_curve = average_over_repetitions(n_repetitions,learning_rate,policy,epsilon,temp,gamma,hidden_layers,\
                            use_er,use_tn,num_iterations,depth,learn_freq,target_update_freq)
        Plot.add_curve(learning_curve,label=r'$\alpha$ = {}'.format(learning_rate))
    Plot.save('Learning_Rates.png')
    
if __name__ == '__main__':
    #ablation()
    learn_rate()
    
