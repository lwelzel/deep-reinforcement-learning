#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Wrapper function for parallel running
#
#######################


import numpy as np
import pathos.multiprocessing as mp
from time import strftime, gmtime
from itertools import repeat


def run_parallel_a2c(env=None, num_epochs=1000, batch_size=60, adjust_factor=0.1, adjust=True, max_reward=200,
                     use_BS=True, use_BLS=True, use_ER=False, use_TES=False, use_AN=False, use_AN_batch=False,
                     buffer_type=None, buffer_depth=1000,
                     epsilon=1., tanh_temp=1.,
                     decay=0.99,
                     batch_decay=0.9,
                     batch_base=0.1,
                     learning_rate=5e-3, discount=0.95, hidden_layers_actor=(32, 16), hidden_act_actor='tanh',
                     kernel_init_actor="glorot_uniform", actor_output_activation="tanh",
                     hidden_layers_critic=(32, 16), hidden_act_critic='relu',
                     kernel_init_critic="glorot_uniform",
                     id=0,
                     repeats=1, load=0.9, sdir="runs/test"):
    """
    Trains many DQN agents in parallel, will use and block load (defaults to 90%) of available cores on the machine
     so that training might significantly slow down other processes.

    If input args are not of type np.array it expects single value (ints, floats, bools etc)
     which it then spans over # av_cores * repeats
     number of cores must be at least two (I think, not sure actually)
    If input args are np.arrays all need to have the same length
    """
    from actor_critic import train_actor_critic

    av_cores = int(load * mp.cpu_count())
    repeats = av_cores

    name = f"a2c_{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}"

    ids = np.arange(repeats)

    print(
        f"Starting pool with {av_cores} cores. That is {av_cores / mp.cpu_count() * 100:.0f}% of all available cores.\n"
        f"These cores are blocked until the run is finished. You can change the percentage via the 'load' option.")
    print(f"\tAll data structures are pre-initialized so that, if the allocated memory is insufficient,\n"
          f"\tthe run will fail during setup.")
    pool = mp.Pool(av_cores)

    # we use map for convenience and we dont really care about returning.
    # Ofc blocking of cores is shitty if they are done, but runtimes should be very similar
    # https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap
    __ = pool.starmap(train_actor_critic,
                      zip(repeat(env, repeats),
                          repeat(num_epochs, repeats),
                          repeat(batch_size, repeats),
                          repeat(adjust_factor, repeats),
                          repeat(adjust, repeats),
                          repeat(max_reward, repeats),
                          repeat(use_BS, repeats),
                          repeat(use_BLS, repeats),
                          repeat(use_ER, repeats),
                          repeat(use_TES, repeats),
                          repeat(use_AN, repeats),
                          repeat(use_AN_batch, repeats),
                          repeat(buffer_type, repeats),
                          repeat(buffer_depth, repeats),
                          repeat(tanh_temp, repeats),
                          repeat(decay, repeats),
                          repeat(batch_decay, repeats),
                          repeat(batch_base, repeats),
                          repeat(learning_rate, repeats),
                          repeat(discount, repeats),
                          repeat(hidden_layers_actor, repeats),
                          repeat(hidden_act_actor, repeats),
                          repeat(kernel_init_actor, repeats),
                          repeat(actor_output_activation, repeats),
                          repeat(hidden_layers_critic, repeats),
                          repeat(hidden_act_critic, repeats),
                          repeat(kernel_init_critic, repeats),
                          repeat(name, repeats), ids,
                          repeat(sdir, repeats)))

    pool.close()
    pool.join()
    return


def main_ac():
    run_parallel_a2c(None, num_epochs=1000, batch_size=60, adjust_factor=0.1, adjust=True, max_reward=200,
                     use_BS=True, use_BLS=True, use_ER=False, use_TES=False, use_AN=False, use_AN_batch=False,
                     buffer_type=None, buffer_depth=1000,
                     epsilon=1., tanh_temp=10.,
                     decay=0.95,
                     learning_rate=5e-3, discount=0.95, hidden_layers_actor=(32, 16), hidden_act_actor='relu',
                     kernel_init_actor="glorot_uniform", actor_output_activation="tanh",
                     hidden_layers_critic=(32, 16), hidden_act_critic='relu',
                     kernel_init_critic="glorot_uniform",
                     id=0,
                     repeats=1, load=0.9, sdir="runs/test")
    return


if __name__ == '__main__':
    main_ac()
