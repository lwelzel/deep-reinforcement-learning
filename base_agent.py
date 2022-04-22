#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Parent class for all training agents 
#
#######################

import tensorflow

class BaseAgent:
    def __init__():
        self.policy = self._create_neural_net()        


    def select_action(self):
        pass;


    def _create_neural_net(self):
        """Neural Network Policy"""

        model = keras.Sequental()
        model.add(keras.Input(shape=(self.n_inputs,)))

        if layers == None:
            print("WARNING: No hidden layers given for Neural Network")
            input("Continue? ... ")
        else:
            for n_nodes in hidden_layers:
                model.add(layers.Dense(n_nodes, activation=hidden_act, kernel_initializer=init))

        model.add(layers.Dense(self.action_space.n, kernel_initializer=init))
        model.summary()
        model.compile()

        return model()

    def save():


    #ANNEALING FUNCTIONS

