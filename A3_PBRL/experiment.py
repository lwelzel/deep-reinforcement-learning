#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Full experiment file to create and plot all
#  results for final report  
#
#######################

import time
from training_functions import train

def avg_over_reps(n_reps=3, **kwargs):
    """
    """
    
    now = time.time()
    for rep in range(n_reps): # Loop over repetitions
        train('reinforce',**kwargs)
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    return


def hyperparam_study():
    """ REINFORCE algorithm hyperparameter study
    """
    
    for method in ["egreedy","softmax"]:
        avg_over_reps(exp_policy=method, name=f"exploration_{method}")
    
    for e in [0.5,0.9,1.0]:
        for d in [0.85, 0.9, 0.999]:
            avg_over_reps(epsilon=e, decay=d, name=f"epsilon={e}_decay={d}")
    
    for gamma in [0.1,0.3,0.6]:
        avg_over_reps(discount=gamma, name=f"discount={gamma}")
    
    for layers in [[64,32],[128,64],[256,128,64]]:
        avg_over_reps(hidden_layers=layers, name=f"layers={layers}")
    return

def main():
    hyperparam_study()


if __name__ == '__main__':
    main()
