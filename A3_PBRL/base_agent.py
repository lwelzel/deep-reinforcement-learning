#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Parent class for all training agents 
#
#######################

import warnings
import h5py
from tensorflow.keras import layers, Input, Sequential
import numpy as np
from helper import softmax, argmax
from time import gmtime, strftime
from pathlib import Path

import gym


class BaseAgent:
    def __init__(self,
                 state_space, action_space,
                 max_reward=500,
                 exp_policy='egreedy', epsilon=1., temperature=1.,
                 anneal_method='exponential',
                 decay=0.999, epsilon_min=0.01, temp_min=0.1,
                 learning_rate=0.01, discount=0.8,
                 hidden_layers=[64, 64], hidden_act='relu', kernel_init=None,
                 name="", id=0):

        self.state_space = state_space.shape
        self.action_space = action_space
        self.n_actions = action_space.n
        self.possible_actions = np.arange(self.n_actions, dtype=int)
        self.arr_prob = np.ones_like(self.possible_actions, dtype=float)
        self.max_reward = max_reward

        self.rng = np.random.default_rng()

        self.exp_policy = exp_policy
        self.epsilon = epsilon
        self.temp = temperature
        self.anneal_method = anneal_method
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon
        self.temp_min = temp_min
        self.temp_max = temperature

        # Exploration and annealing schedule selection
        if exp_policy == "egreedy":
            self.select_action = self.select_action_egreedy
            if anneal_method == "exponential":
                self.anneal_policy_parameter = self.anneal_egreedy_exponential
            elif anneal_method == "linear":
                self.anneal_policy_parameter = self.anneal_egreedy_linear
            else:
                self.anneal_policy_parameter = self.anneal_null
        elif exp_policy == "softmax":
            self.select_action = self.select_action_softmax
            if anneal_method == "exponential":
                self.anneal_policy_parameter = self.anneal_softmax_exponential
            elif anneal_method == "linear":
                self.anneal_policy_parameter = self.anneal_softmax_linear
            else:
                self.anneal_policy_parameter = self.anneal_null
        else:
            print("Policy defaulted to e-greedy.\n"
                  "Anneal defaulted to exponential.")
            self.select_action = self.select_action_egreedy
            self.anneal_policy_parameter = self.anneal_egreedy_exponential

        self.learning_rate = learning_rate
        self.discount = discount

        self.network = self._create_neural_net(hidden_layers, hidden_act, kernel_init)

        self.agent_name = f"run{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}"
        self.dir = Path(f"{name}_a={learning_rate}_g={discount}_{exp_policy}_{anneal_method}_id={id}")
        print(self.dir)

    def select_action_egreedy(self, s):
        actions = self.network.predict(s.reshape((1, 4)))[0]

        try:
            self.arr_prob[:] = self.epsilon / self.n_actions
            self.arr_prob[argmax(actions)] += (1 - self.epsilon)
        except KeyError:
            raise KeyError("No epsilon given in select_action().")

        try:
            a = self.rng.choice(self.possible_actions, None, p=self.arr_prob)
        except ValueError:
            # some of the annealers are a *little* unstable sometimes, fall back on greedy
            warnings.warn("Invalid value due to annealing scheduler. Falling back on greedy policy for this step.")
            return argmax(self.arr_prob)

        return a

    def select_action_softmax(self, s):
        actions = self.network.predict(s.reshape((1, 4)))[0]
        try:
            self.arr_prob[:] = softmax(actions, self.temp)
        except KeyError:
            raise KeyError("No temperature supplied in select_action().")

        try:
            a = self.rng.choice(self.possible_actions, None, p=self.arr_prob)
        except ValueError:
            # some of the annealers are a *little* unstable sometimes, fall back on greedy
            warnings.warn("Invalid value due to annealing scheduler. Falling back on greedy policy for this step.")
            return argmax(self.arr_prob)

        return a

    def anneal_null(self, *args):
        pass

    def anneal_egreedy_exponential(self, *args):
        self.epsilon = np.clip(self.epsilon * self.decay, a_min=self.epsilon_min, a_max=1.)

    def anneal_egreedy_linear(self, t, t_final):
        self.epsilon = self.linear_anneal(t, t_final, self.epsilon_max, self.epsilon_min)

    def anneal_softmax_exponential(self, *args):
        self.temp *= self.decay

    def anneal_softmax_linear(self, t, t_final):
        self.temp = self.linear_anneal(t, t_final, self.temp_min, self.temp_max)

    def _create_neural_net(self, hidden_layers, hidden_act, kernel_init):
        """Neural Network Policy"""

        model = Sequential()
        model.add(Input(shape=self.state_space))

        if layers is None:
            print("WARNING: No hidden layers given for Neural Network")
            input("Continue? ... ")
        else:
            for n_nodes in hidden_layers:
                model.add(layers.Dense(n_nodes, activation=hidden_act, kernel_initializer=kernel_init))

        model.add(layers.Dense(self.n_actions, kernel_initializer=kernel_init))
        model.summary()
        model.compile()

        return model

    def update_weights(self, weight_grad):
        # Not sure if weights and weight_grad are in same shapes. CHECK
        weights = self.network.get_weights
        new_weights = weights + self.learning_rate * weight_grad
        self.network.set_weights(new_weights)

    def save(self, rewards):
        """Saves rewards list to directory"""
        rewards = rewards[np.isfinite(rewards)]

        Path(self.dir).mkdir(parents=True, exist_ok=True)

        try:
            f = h5py.File(self.dir / "Rewards_{}.h5".format(self.agent_name), 'w')
            f.create_dataset("rewards", data=rewards)

            ### save simulation data to h5 file
            meta_dict = {"alpha": self.learning_rate,
                         "gamma": self.discount,
                         "policy": self.exp_policy,
                         "anneal": self.anneal_method,
                         "max_reward": self.max_reward,
                         }

            # Store metadata in hdf5 file
            for k in meta_dict.keys():
                f.attrs[k] = str(meta_dict[k])
            f.close()
        except BaseException:
            print(f"!! a file could not be saved !!")

    @staticmethod
    def linear_anneal(t, t_end, start, final):
        """ Linear annealing scheduler
        t: current timestep
        T: total timesteps
        start: initial value
        final: value after percentage*T steps
        percentage: percentage of T after which annealing finishes
        """

        return start + (start - final) * (t_end - t) / t_end


def main():
    env = gym.make('CartPole-v1')
    pi = BaseAgent(env.observation_space, env.action_space)


if __name__ == '__main__':
    main()
