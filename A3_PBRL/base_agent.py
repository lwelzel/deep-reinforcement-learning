#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Parent class for all training agents 
#
#######################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Sequential
from helper import softmax, argmax

import gym

class BaseAgent:
    def __init__(self, 
                state_space, action_space,
                exp_policy='egreedy', epsilon=1., temperature=1.,
                anneal_method = 'exponential',
                learning_rate=0.01, discount=0.8,
                hidden_layers=[64,64], hidden_act='relu', kernel_init=None):

        self.state_space    = state_space.shape
        self.action_space   = action_space
        self.n_actions      = action_space.n
        self.possible_actions= np.arange(self.n_actions, dtype=int)
        self.arr_prob       = np.ones_like(self.possible_acts, dtype=float)        

        self.rng = np.random.default_rng()

        self.exp_policy     = exp_policy
        self.epsilon        = epsilon
        self.temp           = temperature
        self.anneal_method  = anneal_method
        
        if exp_policy == 'egreedy':
            self.select_action = self.select_action_egreedy
        elif exp_policy == 'softmax':
            self.select_action = self.select_action_softmax
        else:
            print(f"Unknown exploration policy provided ({exp_policy}).\n"
                    "Defaulted to e-greedy") 
            self.select_action = self.select_action_egreedy                     
        

        self.learning_rate  = learning_rate
        self.discount       = discount

        self.network = self._create_neural_net(hidden_layers, hidden_act, kernel_init)        


    def select_action_egreedy(self, s):
        actions = self.network.predict(s.reshape((1,4)))[0]
        
        try:
            self.arr_prob[:] = self.epsilon / self.n_actions
            self.arr_prob[argmax(actions)] += (1-self.epsilon)
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

          

    def _create_neural_net(self, hidden_layers, hidden_act, kernel_init):
        """Neural Network Policy"""

        model = Sequential()
        model.add(Input(shape=self.state_space))

        if layers == None:
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
        #Not sure if weights and weight_grad are in same shapes. CHECK
        weights = self.network.get_weights
        new_weights = weights + self.learning_rate * weight_grad
        self.network.set_weights(new_weights)


    def save(self):
        pass;


def main():
    env = gym.make('CartPole-v1')
    pi = BaseAgent(env.observation_space,env.action_space)

if __name__ == '__main__':
    main()

