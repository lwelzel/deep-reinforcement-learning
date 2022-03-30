import numpy as np
from dqn import learn_dqn, wrapper_dqn_learn_save
import pathos.multiprocessing as mp

# mp required pre-imports:
from pathlib import Path
import time
from time import perf_counter, strftime, gmtime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import gym
from buffer_class import MetaBuffer
from helper import LearningCurvePlot, smooth, softmax, argmax


def run_parallel_dqns(learning_rate, policy, epsilon, temp,
                      gamma, hidden_layers, use_er, use_tn,
                      num_iterations, depth=2500, learn_freq=4,
                      target_update_freq=25, sample_batch_size=128,
                      anneal_method=None, render=False,
                      repeats=1, load=0.9):
    """
    Trains many DQN agents in parallel, will use and block load (defaults to 90%) of available cores on the machine
     so that training might significantly slow down other processes.

    If input args are not of type np.array it expects single value (ints, floats, bools etc)
     which it then spans over # av_cores * repeats
     number of cores must be at least two
    If input args are np.arrays all need to have the same length
    """
    # av_cores = int(load * mp.cpu_count())
    av_cores = 2

    if not isinstance(learning_rate, np.ndarray):
        learning_rate = np.repeat(learning_rate, av_cores * repeats)
        policy = np.repeat(policy, av_cores * repeats)
        epsilon = np.repeat(epsilon, av_cores * repeats)
        temp = np.repeat(temp, av_cores * repeats)
        gamma = np.repeat(gamma, av_cores * repeats)
        hidden_layers = np.repeat(hidden_layers, av_cores * repeats)
        use_er = np.repeat(use_er, av_cores * repeats)
        use_tn = np.repeat(use_tn, av_cores * repeats)
        num_iterations = np.repeat(num_iterations, av_cores * repeats)
        depth = np.repeat(depth, av_cores * repeats)
        learn_freq = np.repeat(learn_freq, av_cores * repeats)
        target_update_freq = np.repeat(target_update_freq, av_cores * repeats)
        sample_batch_size = np.repeat(sample_batch_size, av_cores * repeats)
        anneal_method = np.repeat(anneal_method, av_cores * repeats)
        render = np.repeat(render, av_cores * repeats)

    for chunk in [learning_rate, policy, epsilon, temp,
                  gamma, hidden_layers, use_er, use_tn,
                  num_iterations, depth, learn_freq,
                  target_update_freq, sample_batch_size,
                  anneal_method, render]:
        print(chunk.shape)

    print(f"Starting pool with {av_cores} cores. That is {av_cores/mp.cpu_count() * 100:.0f}% of all cores.\n"
          f"These cores are blocked until the run is finished. You can change the percentage via the 'load' option.")
    pool = mp.Pool(av_cores)
    # we use map for convenience and we dont really care about returning.
    # Ofc blocking of cores is shitty if they are done, but runtimes should be very similar
    # https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap
    __ = pool.starmap(learn_dqn,
                  zip(learning_rate, policy, epsilon, temp,
                  gamma, hidden_layers, use_er, use_tn,
                  num_iterations, depth, learn_freq,
                  target_update_freq, sample_batch_size,
                  anneal_method, render))
    return


def main():
    run_parallel_dqns(learning_rate=0.01, policy='egreedy', epsilon=0.1, temp=1.,
                      gamma=1., hidden_layers=[12,6], use_er=True, use_tn=True,
                      num_iterations=25, depth=2500, learn_freq=4,
                      target_update_freq=25, sample_batch_size=128,
                      anneal_method=None, render=False,
                      repeats=1, load=0.9)
    return


if __name__ == '__main__':
    main()
