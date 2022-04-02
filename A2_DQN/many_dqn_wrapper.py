import numpy as np
from dqn_base_class import run
import pathos.multiprocessing as mp
from time import strftime, gmtime

def run_parallel_dqns(num_epochs=50, max_epoch_env_steps=50, target_update_freq=5,
                      policy="egreedy",
                      learning_rate=0.01, gamma=0.8,
                      epsilon=0.5, temperature=1.,
                      hidden_layers=[32, 32], hidden_act='relu', kernel_init='HeUniform',
                      loss_func='mean_squared_error',
                      use_tn=True, use_er=True,
                      buffer_type=None, buffer_depth=2500, sample_batch_size=100,
                      id=0,
                      repeats=1, load=0.9):
    """
    Trains many DQN agents in parallel, will use and block load (defaults to 90%) of available cores on the machine
     so that training might significantly slow down other processes.

    If input args are not of type np.array it expects single value (ints, floats, bools etc)
     which it then spans over # av_cores * repeats
     number of cores must be at least two (I think, not sure actually)
    If input args are np.arrays all need to have the same length
    """
    av_cores = int(load * mp.cpu_count())

    name = f"{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}"

    # would probably be easier with repeat :(
    if not isinstance(num_epochs, np.ndarray):
        num_epochs = np.repeat(num_epochs, av_cores * repeats)
        max_epoch_env_steps = np.repeat(max_epoch_env_steps, av_cores * repeats)
        target_update_freq = np.repeat(target_update_freq, av_cores * repeats)
        policy = np.repeat(policy, av_cores * repeats)
        learning_rate = np.repeat(learning_rate, av_cores * repeats)
        gamma = np.repeat(gamma, av_cores * repeats)
        epsilon = np.repeat(epsilon, av_cores * repeats)
        temperature = np.repeat(temperature, av_cores * repeats)
        hidden_layers = np.tile(hidden_layers, av_cores * repeats).reshape((-1, len(hidden_layers)))
        hidden_act = np.repeat(hidden_act, av_cores * repeats)
        kernel_init = np.repeat(kernel_init, av_cores * repeats)
        loss_func = np.repeat(loss_func, av_cores * repeats)
        use_tn = np.repeat(use_tn, av_cores * repeats)
        use_er = np.repeat(use_er, av_cores * repeats)
        buffer_type = np.repeat(buffer_type, av_cores * repeats)
        buffer_depth = np.repeat(buffer_depth, av_cores * repeats)
        sample_batch_size = np.repeat(sample_batch_size, av_cores * repeats)
        name = np.repeat(name, av_cores * repeats)

    ids = np.arange(len(learning_rate))

    print(
        f"Starting pool with {av_cores} cores. That is {av_cores / mp.cpu_count() * 100:.0f}% of all available cores.\n"
        f"These cores are blocked until the run is finished. You can change the percentage via the 'load' option.")
    print(f"\tAll data structures are pre-initialized so that, if the allocated memory is insufficient,\n"
          f"\tthe run will fail during setup.")
    pool = mp.Pool(av_cores)

    # we use map for convenience and we dont really care about returning.
    # Ofc blocking of cores is shitty if they are done, but runtimes should be very similar
    # https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap
    __ = pool.starmap(run,
                      zip(num_epochs, max_epoch_env_steps, target_update_freq,
                          policy,
                          learning_rate, gamma,
                          epsilon, temperature,
                          hidden_layers, hidden_act, kernel_init,
                          loss_func,
                          use_tn, use_er,
                          buffer_type, buffer_depth, sample_batch_size,
                          name, ids))

    pool.close()
    pool.join()
    return


def main():
    run_parallel_dqns(num_epochs=500, max_epoch_env_steps=200, target_update_freq=4,
                      policy="egreedy",
                      learning_rate=0.01, gamma=0.9,
                      epsilon=0.9, temperature=1.,
                      hidden_layers=[512, 256, 64], hidden_act='relu', kernel_init='HeUniform',
                      loss_func='mean_squared_error',
                      use_tn=True, use_er=True,
                      buffer_type=None, buffer_depth=2000, sample_batch_size=64,
                      id=0,
                      repeats=1, load=0.9)
    return


if __name__ == '__main__':
    main()
