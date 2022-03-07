#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
import time
from tqdm import tqdm
from Q_learning import q_learning, q_learning_anneal
from SARSA import sarsa
from MonteCarlo import monte_carlo
from Nstep import n_step_Q
from Helper import LearningCurvePlot, smooth, AnnealScheduler
import h5py
import DP_evaluations

def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', 
                    epsilon=None, temp=None, smoothing_window=51, plot=False, n=5,
                             anneal="linear_anneal",
                             buffer=25, start=0.3, final=0.0, Q_thresh=0.05, r_thresh=0.05, percentage=0.5):

    reward_results = np.empty([n_repetitions,n_timesteps]) # Result array
    now = time.time()
    
    for rep in tqdm(range(n_repetitions), leave=False): # Loop over repetitions
        if backup == 'q':
            rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
        elif backup == 'aq':
            rewards = q_learning_anneal(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot,
                                        anneal_schedule=anneal,
                                        buffer=buffer, start=start, final=final, Q_thresh=Q_thresh,
                                        r_thresh=r_thresh, percentage=percentage
                                        )
        elif backup == 'sarsa':
            rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
        elif backup == 'mc':
            rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
        elif backup == 'nstep':
            rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)

        reward_results[rep] = rewards
        
    # print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve, smoothing_window) # additional smoothing
    return learning_curve

def save_run(file, data_name, data):
    file.create_dataset(data_name,
                        compression="gzip",
                        data=data)

def experiment():
    print("\nDRL Assignment 1: TRL by Lukas Welzel.\n"
          "\n"
          "\tIn case something doesnt work please see the appendix first.\n"
          "\tLong runs with annealing might raise warnings. Those should be handled internally.\n"
          "\tIf they are not handled internally you found a bug that got through testing and are allowed to re-run.\n"
          "\tThis is probably bad luck like a propagating division by zero (nan) that was not caught.\n"
          "\n"
          "\tRuntime for the Bonus questions is rather high on my machine (20+ min) with 50 reps and 50k total steps.\n"
          "\tI added some tqdm progress bars but none of them are global, sorry.\n"
          "\tAlso, if you have issues with flashing lights you should probably not use very small n_timesteps due to the progres bars.\n\n")
    ####### Settings
    # Experiment
    # TODO:
    n_repetitions = 2  # 50
    smoothing_window = 1001
    file = h5py.File("runs_normal.h5", "w")

    # MDP
    # TODO:
    n_timesteps = 1000  # 50000
    max_episode_length = 100
    gamma = 1.0

    print(f"\tRunning each trial for {n_repetitions} repetitions and {n_timesteps} total steps in each repetition.\n"
          f"\tRoughly {n_repetitions * n_timesteps * 23:.1e} steps in total. Good luck!\n\n")

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.05
    temp = 1.0
    
    # Target and update
    backup = 'q' # 'q' or 'sarsa' or 'mc' or 'nstep'
    learning_rate = 0.25
    n = 5
        
    # Plotting parameters
    plot = False
    
    # Nice labels for plotting
    policy_labels = {'egreedy': '$\epsilon$-greedy policy',
                     'softmax': 'Softmax policy',
                     'aegreedy': 'annealing $\epsilon$-greedy policy',
                     'asoftmax': 'annealing Softmax policy'
                     }

    backup_labels = {'q': 'Q-learning',
                  'sarsa': 'SARSA',
                  'mc': 'Monte Carlo',
                  'nstep': 'n-step Q-learning'}
    
    ####### Experiments

    # Bonus stuff 0
    DP_evaluations.q_val_vis()
    scheduler = AnnealScheduler()
    scheduler.show()
    
    #### Assignment 1: Dynamic Programming
    # Execute this assignment in DynamicProgramming.py
    optimal_average_reward_per_timestep = 1.034 # set the optimal average reward per timestep you found in the DP assignment here
    
    #### Assignment 2: Effect of exploration
    backup = 'q'
    Plot = LearningCurvePlot(title='Q-learning: effect of $\epsilon$-greedy versus softmax exploration')
    # the ICML standard does not allow titles

    policy = 'egreedy'
    epsilons = [0.01, 0.05, 0.2]
    for epsilon in tqdm(epsilons, leave=True, desc="QL, egreedy"):
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))
        try:
            save_run(file, backup + f'e-greedy e={epsilon:.0e}', learning_curve)
        except (ValueError, BaseException):
            pass

    policy = 'softmax'
    temps = [0.01, 0.1, 1.0]
    for temp in tqdm(temps, leave=True, desc="QL, softmax"):
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))
        try:
            save_run(file, backup + f'softmax t={temp:.0e}', learning_curve)
        except (ValueError, BaseException):
            pass


    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('exploration.png')
    policy = 'egreedy'
    epsilon = 0.05 # set epsilon back to original value
    temp = 1.0

    
    ##### Assignment 3: Q-learning versus SARSA
    backups = ['q','sarsa']
    learning_rates = [0.05,0.2,0.4]
    Plot = LearningCurvePlot(title = 'Q-learning versus SARSA')
    for backup in backups:
        for learning_rate in tqdm(learning_rates, leave=True, desc=f"{backup}, vary learning rates"):
            learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                                  gamma, policy, epsilon, temp, smoothing_window, plot, n)
            Plot.add_curve(learning_curve,label=r'{}, $\alpha$ = {} '.format(backup_labels[backup],learning_rate))
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('on_off_policy.png')
    # Set back to original values
    learning_rate = 0.25
    backup = 'q'


    ##### Assignment 4: Back-up depth
    backup = 'nstep'
    ns = [1,3,5,10,20,100]
    # ns = [1,  5, 100]
    Plot = LearningCurvePlot(title = 'Effect of target depth')
    for n in tqdm(ns, leave=True, desc="n-Step, egreedy"):
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'{}-step Q-learning'.format(n))
    backup = 'mc'
    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                          gamma, policy, epsilon, temp, smoothing_window, plot, n)
    Plot.add_curve(learning_curve,label='Monte Carlo')
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('depth.png')

    # ##### Bonus 1: Annealing exploration
    backup = 'q'
    Plot = LearningCurvePlot(title='Q-learning: annealing $\epsilon$-greedy and softmax exploration')
    # the ICML standard does not allow titles

    policy = 'softmax'
    temps = [0.01]
    for temp in tqdm(temps, leave=True, desc="QL, softmax"):
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))
        try:
            save_run(file, backup + f'softmax t={temp:.0e}', learning_curve)
        except (ValueError, BaseException):
            pass

    policy = 'egreedy'
    epsilons = [0.01]
    for epsilon in tqdm(epsilons, leave=True, desc="QL, egreedy"):
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))
        try:
            save_run(file, backup + f'e-greedy e={epsilon:.0e}', learning_curve)
        except (ValueError, BaseException):
            pass

    backup = "aq"
    policy = 'egreedy'
    anneals = [
        "q_error_anneal",
        "r_diff_anneal",
        "linear_anneal",
        "logistic_anneal"
    ]
    for anneal in tqdm(anneals, leave=True, desc="annealing QL, a-egreedy"):
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n,
                                                  anneal=anneal,
                                                  buffer=10, start=0.3, final=0.0, Q_thresh=0.02, r_thresh=0.02,
                                                  percentage=0.5
                                                  )
        Plot.add_curve(learning_curve,label=r'annealing $\epsilon$-greedy, = {}'.format(anneal))
        try:
            save_run(file, backup + f'a-e-greedy a={anneal}', learning_curve)
        except (ValueError, BaseException):
            pass



    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('ql_anneal.png')
    policy = 'egreedy'
    epsilon = 0.05 # set epsilon back to original value
    temp = 1.0

    print("\nDone.\n\n")


if __name__ == '__main__':
    experiment()