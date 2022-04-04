# author: lwelzel, adapted from initial work by rvit, bslik
# ! /usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import time
from tqdm import tqdm
from time import perf_counter, strftime, gmtime, time, sleep
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers  # , Sequential, Input
import gym
from buffer_class import MetaBuffer, PrioBuffer
from helper import LearningCurvePlot, smooth, softmax, argmax
import h5py
import os
import warnings

# CUDA settings: one gpu, variable memory to avoid blocking of all memory (on GPU)
try:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except BaseException:
    pass

import_time = f"{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}"


class DQNAgent:
    def __init__(self,
                 state_space, action_space,
                 policy="egreedy",
                 learning_rate=0.01, gamma=0.8,
                 epsilon=1., temperature=1.,
                 hidden_layers=None, hidden_act='relu', kernel_init='HeUniform', loss_func='mean_squared_error',
                 use_tn=False, use_er=False,
                 anneal="exponential",
                 smart_update=False, soft_update=False,
                 buffer_type=None, buffer_depth=2500, sample_batch_size=100,
                 name="placeholder", id=0):

        self.state_space = state_space.shape
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy = policy
        self.epsilon_initial = epsilon
        self.epsilon = epsilon
        self.epsilon_max = epsilon
        self.epsilon_min = 0.01
        self.decay = 0.999
        self.temperature = temperature
        self.anneal = anneal

        if policy == "egreedy":
            self.select_action = self.select_action_egreedy
            if anneal == "exponential":
                self.anneal_policy_parameter = self.anneal_egreedy_exponential
            elif anneal == "linear":
                self.anneal_policy_parameter = self.anneal_egreedy_linear
            else:
                self.anneal_policy_parameter = self.anneal_null
        elif policy == "softmax":
            self.select_action = self.select_action_softmax
            if anneal == "exponential":
                self.anneal_policy_parameter = self.anneal_softmax_exponential
            elif anneal == "linear":
                self.anneal_policy_parameter = self.anneal_softmax_linear
            else:
                self.anneal_policy_parameter = self.anneal_null
        else:
            print("Policy defaulted to e-greedy.\n"
                  "Anneal defaulted to exponential.")
            self.select_action = self.select_action_egreedy
            self.anneal_policy_parameter = self.anneal_egreedy_exponential

        self.hidden_layers = hidden_layers
        self.hidden_act = hidden_act
        self.kernel_init = kernel_init
        self.loss_func = loss_func

        self.rng = np.random.default_rng()

        self.action_space = action_space
        self.n_actions = float(action_space.n)
        self.possible_actions = np.arange(action_space.n, dtype=int)
        self.arr_prob = np.ones_like(self.possible_actions, dtype=float)

        self.use_tn = use_tn
        self.use_er = use_er

        self.smart_update = smart_update
        self.soft_update = soft_update
        self.soft_tau_min = 0.1
        self.soft_tau_max = 0.5

        self.online_DQN_network = self.make_model()
        if use_tn:
            self.target_DQN_network = self.make_model()
        else:
            self.target_DQN_network = None  # self.online_DQN_network

        self.buffer_type = buffer_type
        if buffer_type is None:
            self.buffer = MetaBuffer(buffer_depth, sample_batch_size)
        elif buffer_type == "priority":
            self.buffer = PrioBuffer(buffer_depth, sample_batch_size)
        else:
            raise KeyError("No valid buffer type provided.")

        self.id = id
        self.agent_name = f"run={strftime('%Y-%m-%d-%H-%M-%S', gmtime())}_id{self.id}"
        self.dir = Path(f"batch={name}_"
                        f"a={self.learning_rate:.0e}_"
                        f"g={self.gamma:.0e}_"
                        f"hlay={tuple(hidden_layers)}")

    def make_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(self.state_space[0],)))

        if layers == None:
            print("WARNING: No hidden layers given for Neural Network")
            input("Continue? ... ")
        else:
            for n_nodes in self.hidden_layers:
                model.add(layers.Dense(n_nodes, activation=self.hidden_act, kernel_initializer=self.kernel_init))

        model.add(layers.Dense(self.action_space.n, kernel_initializer=self.kernel_init))
        # model.compile(loss=self.loss_func, optimizer=optimizers.Adam(0.001))
        model.compile(loss=self.loss_func,
                      optimizer=optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                      metrics=["accuracy"])

        # model.summary()
        return model

    def replay(self):
        # s(t), a(t), r(t), s(t+1), done(t+1)
        # print(self.buffer._buffer)
        # print(self.buffer.sample)
        states, actions, rewards, states_next, dones = self.buffer.sample
        actions = actions.astype(int).reshape((-1, 1))

        # print(states, actions, rewards, states_next, dones)

        target = self.online_DQN_network.predict(states)

        if self.use_tn:  # DDQN, select one option
            # == DDQN option 1: direct prediction using TN ==
            # massive mistake in the function below, whoever spots it gets a cookie
            # so lucky we never used it because it was unstable (I wonder why :D)
            # very unstable, fast

            # target_next = self.target_DQN_network.predict(states_next)
            # values = (rewards + (1 - dones) * self.gamma * np.take_along_axis(target_next,
            #                                                                   actions,
            #                                                                   axis=1)).reshape((-1, 1))
            # np.put_along_axis(target, actions, values, axis=1)

            # == DDQN option 2: indirect prediction using TN via online network targets ==
            # more stable, slower

            target_next = self.online_DQN_network.predict(states_next)
            target_target = self.target_DQN_network.predict(states_next)

            online_target_actions = np.argmax(target_next, axis=1).reshape((-1, 1))
            values = (rewards + (1 - dones) * self.gamma * np.take_along_axis(target_target,
                                                                              online_target_actions,
                                                                              axis=1)).reshape((-1, 1))
            np.put_along_axis(target, actions, values, axis=1)

        else:  # basic DQN
            target_next = self.online_DQN_network.predict(states_next)
            values = rewards + (1 - dones) * self.gamma * np.max(target_next, axis=1).reshape((-1, 1))
            np.put_along_axis(target, actions, values, axis=1)




        self.online_DQN_network.fit(states, target,
                                    batch_size=self.buffer.depth, verbose=0)

    def update_target_network(self, reward=None, max_reward=None):
        if self.smart_update:
            if self.soft_update:
                online_weights = self.online_DQN_network.get_weights()
                target_weights = self.target_DQN_network.get_weights()

                tau = 0.1

                weights = np.array([target_weight * (1. - tau) + online_weight * tau
                                    for online_weight, target_weight
                                    in zip(online_weights, target_weights)],
                                   dtype=object)
                self.target_DQN_network.set_weights(weights)

            else:  # homebrew
                online_weights = self.online_DQN_network.get_weights()
                target_weights = self.target_DQN_network.get_weights()

                tau = self.linear_anneal(reward, max_reward, self.soft_tau_max,  self.soft_tau_min)

                weights = np.array([target_weight * (1. - tau) + online_weight * tau
                                    for online_weight, target_weight
                                    in zip(online_weights, target_weights)],
                                   dtype=object)
                self.target_DQN_network.set_weights(weights)

        else:
            self.target_DQN_network.set_weights(self.online_DQN_network.get_weights())

    def anneal_null(self, *args):
        pass

    def anneal_egreedy_exponential(self, *args):
        self.epsilon = np.clip(self.epsilon * self.decay, a_min=self.epsilon_min, a_max=1.)

    def anneal_egreedy_linear(self, t, t_final):
        self.epsilon = self.linear_anneal(t, t_final, self.epsilon_max, self.epsilon_min)

    def anneal_softmax_exponential(self, *args):
        self.temperature *= self.decay

    def anneal_softmax_linear(self, t, t_final):
        self.temperature = self.linear_anneal(t, t_final, self.soft_tau_min, self.soft_tau_max)

    def select_action_egreedy(self, s):
        actions = self.online_DQN_network.predict(s.reshape((1, 4)))[0]

        try:
            # should precalculate this and then draw from matrix
            self.arr_prob[:] = self.epsilon / self.n_actions
            self.arr_prob[argmax(actions)] = 1 - self.epsilon * ((self.n_actions - 1.)
                                                                 / self.n_actions)
            # TO DO: Add own code
        except KeyError:
            raise KeyError("No epsilon supplied in select_action().")

        try:
            a = self.rng.choice(self.possible_actions, None, p=self.arr_prob)
        except ValueError:
            # some of the annealers are a *little* unstable sometimes, fall back on greedy
            warnings.warn("Invalid value due to annealing scheduler. Falling back on greedy policy for this step.")
            return argmax(self.arr_prob)

        return a

    def select_action_softmax(self, s):
        actions = self.target_DQN_network.predict(s.reshape((1, 4)))[0]
        try:
            self.arr_prob[:] = softmax(actions, self.temperature)
        except KeyError:
            raise KeyError("No temperature supplied in select_action().")

        try:
            a = self.rng.choice(self.possible_actions, None, p=self.arr_prob)
        except ValueError:
            # some of the annealers are a *little* unstable sometimes, fall back on greedy
            warnings.warn("Invalid value due to annealing scheduler. Falling back on greedy policy for this step.")
            return argmax(self.arr_prob)

        return a

    def save(self, rewards, max_reward):
        rewards = rewards[np.isfinite(rewards)]
        """Saves Deep Q-Network and array of rewards"""
        Path(self.dir).mkdir(parents=True, exist_ok=True)


        try:
            self.online_DQN_network.save(self.dir / "DeepQN_model_{}.h5".format(self.agent_name))
            f = h5py.File(self.dir / "Rewards_{}.h5".format(self.agent_name), 'w')
            f.create_dataset("rewards", data=rewards)

            ### save simulation data to h5 file
            meta_dict = {"alpha": self.learning_rate,
                         "gamma": self.gamma,
                         "buffer_shape": self.buffer._buffer.shape,
                         "buffer_method": self.buffer_type,
                         "buffer_sample": self.buffer._sample_batch_length,
                         "policy": self.policy,
                         "method_para": self.epsilon_initial,
                         "anneal": self.anneal,
                         "max_reward": max_reward,
                         "use_TN": str(self.use_tn),
                         "use_ER": str(self.use_er),
                         }

            # Store metadata in hdf5 file
            for k in meta_dict.keys():
                f.attrs[k] = str(meta_dict[k])
            f.close()
        except BaseException:
            # this is the first time I use recursion as a safety feature, wow!
            print(f"!! a file could not be saved !!")

    @staticmethod
    def linear_anneal(t, t_end, start, final):
        ''' Linear annealing scheduler
        t: current timestep
        T: total timesteps
        start: initial value
        final: value after percentage*T steps
        percentage: percentage of T after which annealing finishes
        '''

        return start + (start - final) * (t_end - t) / t_end


def run(num_epochs, max_epoch_env_steps, target_update_freq,
        policy="egreedy",
        learning_rate=0.01, gamma=0.8,
        epsilon=0.5, temperature=1.,
        hidden_layers=None, hidden_act='relu', kernel_init='HeUniform', loss_func='mean_squared_error',
        use_tn=False, use_er=False,
        anneal="exponential",
        smart_update=False, soft_update=False,
        buffer_type=None, buffer_depth=2500, sample_batch_size=100,
        name="placeholder", id=0):
    # set buffer_depth, sample_batch_size to 1 for DQN without ER
    maxtime = 60. * 60 * 24
    env = gym.make('CartPole-v1')

    state_space = env.observation_space
    action_space = env.action_space

    if not use_er:
        buffer_depth=1
        sample_batch_size=1
        print("Using no ER is equivalent to a buffer depth (capacity) and sampling size of 1.")

    pi = DQNAgent(state_space, action_space,
                  policy=policy,
                  learning_rate=learning_rate, gamma=gamma,
                  epsilon=epsilon, temperature=temperature,
                  hidden_layers=hidden_layers, hidden_act=hidden_act, kernel_init=kernel_init, loss_func=loss_func,
                  use_tn=use_tn, use_er=use_er,
                  anneal=anneal,
                  smart_update=smart_update, soft_update=soft_update,
                  buffer_type=buffer_type, buffer_depth=buffer_depth, sample_batch_size=sample_batch_size,
                  name=name, id=id)

    env._max_episode_steps = max_epoch_env_steps
    epoch_timesteps = np.arange(max_epoch_env_steps)

    rewards = np.full(shape=num_epochs, fill_value=np.nan,
                      dtype=np.float64)

    if use_er:
        done = False
        s = env.reset()
        for timestep in np.arange(buffer_depth + 1):  # +1 to make sure buffer is filled
            if done:
                s = env.reset()
            a = pi.select_action(s)
            s_next, r, done, _ = env.step(a)
            pi.buffer.update_buffer((s, a, r, s_next, done))

    save_reps = 5
    start_time = time()

    if id == 0:
        iterator = tqdm(np.arange(num_epochs), leave=False)
    else:
        iterator = np.arange(num_epochs)

    for epoch in iterator:
        s = env.reset()
        done = False
        rewards[epoch] = 0.
        i=0

        while not done:
            a = pi.select_action(s)
            s_next, r, done, _ = env.step(a)
            rewards[epoch] += r

            r_incentive = np.logical_or(not done, i == env._max_episode_steps - 1)
            r = r_incentive * r + (1 - r_incentive) * - 0.5 * env._max_episode_steps
            i += 1

            # if use_er:
            #     pi.buffer.update_buffer((s, a, r, s_next, done))
            pi.buffer.update_buffer((s, a, r, s_next, done))
            s = s_next

            if pi.use_tn and done:  # and epoch % target_update_freq == 0:
                pi.update_target_network(rewards[epoch], max_epoch_env_steps)

            pi.anneal_policy_parameter(epoch, num_epochs)

            pi.replay()

        if epoch % save_reps == 0.:
            pi.save(rewards, env._max_episode_steps)
            if (time() - start_time) > maxtime:
                print(
                    f"Maximum time exceeded. Stopped learning. Elapsed time: {(time() - start_time) / 60.:.1f} minutes.")
                break

    # save model and learning curve
    pi.save(rewards, env._max_episode_steps)
    env.close()

    return


def test_run():
    num_epochs, max_epoch_env_steps, target_update_freq = 100, 500, 5
    policy = "egreedy"
    learning_rate = 0.01
    gamma = 0.95
    epsilon = 1.
    temperature = 1.
    hidden_layers = [512, 256, 64]
    hidden_act = 'relu'
    kernel_init = 'HeUniform'
    loss_func = 'mean_squared_error'
    use_tn = True
    use_er = True
    anneal = "exponential"
    buffer_type = None
    # set buffer_depth, sample_batch_size to 1 for DQN without ER
    buffer_depth = 1000
    sample_batch_size = 60
    name = f"{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}"
    id = 0


    maxtime = 60. * 60 * 24
    env = gym.make('CartPole-v1')

    state_space = env.observation_space
    action_space = env.action_space

    pi = DQNAgent(state_space, action_space,
                  policy=policy,
                  learning_rate=learning_rate, gamma=gamma,
                  epsilon=epsilon, temperature=temperature,
                  hidden_layers=hidden_layers, hidden_act=hidden_act, kernel_init=kernel_init, loss_func=loss_func,
                  use_tn=use_tn, use_er=use_er,
                  anneal=anneal,
                  buffer_type=buffer_type, buffer_depth=buffer_depth, sample_batch_size=sample_batch_size,
                  name=name, id=id)

    env._max_episode_steps = max_epoch_env_steps
    epoch_timesteps = np.arange(max_epoch_env_steps)

    rewards = np.full(shape=num_epochs, fill_value=np.nan,
                      dtype=np.float64)

    if use_er:
        done = False
        s = env.reset()
        for timestep in tqdm(np.arange(buffer_depth + 1)):  # +1 to make sure buffer is filled
            if done:
                s = env.reset()
            a = pi.select_action(s)
            s_next, r, done, _ = env.step(a)
            pi.buffer.update_buffer((s, a, r, s_next, done))

    print()

    save_reps = int(0.25 * num_epochs)
    start_time = time()

    for epoch in np.arange(num_epochs):
        s = env.reset()
        done = False
        rewards[epoch] = 0.
        i = 0

        while not done:
            a = pi.select_action(s)
            s_next, r, done, _ = env.step(a)
            rewards[epoch] += r

            r_incentive = np.logical_or(not done, i == env._max_episode_steps - 1)
            r = r_incentive * r + (1 - r_incentive) * - 0.5 * env._max_episode_steps
            i += 1

            if use_er:
                pi.buffer.update_buffer((s, a, r, s_next, done))
            s = s_next

            if pi.use_tn and done:  # and epoch % target_update_freq == 0:
                pi.update_target_network(rewards[epoch], max_epoch_env_steps)

            pi.anneal_policy_parameter(epoch, num_epochs)

            pi.replay()

        print(f"{epoch:03.0f}/{num_epochs:03.0f} \t rewards: {rewards[epoch]:03.0f} \t epsilon: {pi.epsilon:01.04f}")

        if epoch % save_reps == 0.:
            pi.save(rewards, env._max_episode_steps)
            if (time() - start_time) > maxtime:
                print(
                    f"Maximum time exceeded. Stopped learning. Elapsed time: {(time() - start_time) / 60.:.1f} minutes.")

        # save model and learning curve
        pi.save(rewards, env._max_episode_steps)
        env.close()


if __name__ == "__main__":
    test_run()
