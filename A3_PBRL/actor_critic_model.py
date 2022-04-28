#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Actor Critic Model Class
#
#######################

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


class ActorCriticModel(Model):
    """
    Subclassed keras.Model to control the passes through the network and customize the layers with more fine control
    """

    def __init__(self, action_space_size, state_space_size,
                 hidden_layers_actor=(258, 64), hidden_act_actor='relu', kernel_init_actor='he_uniform',
                 hidden_layers_critic=(258, 64), hidden_act_critic='relu', kernel_init_critic='he_uniform',
                 policy="softmax"):
        """
        We use 'softmax' as the activation function for the actor output layer
        so that it can receive non-normalized (+/- inf) input from prior layers
        """

        self.action_space_size = action_space_size
        self.state_space_size = state_space_size

        input = Input(shape=(self.state_space_size,), name='input')

        # actor network branch
        self._actor = Dense(units=hidden_layers_actor[0],
                            activation=hidden_act_actor,
                            kernel_initializer=kernel_init_actor)(input)

        for i, neurons in enumerate(hidden_layers_actor[1:]):
            self._actor = Dense(units=neurons,
                                activation=hidden_act_actor,
                                kernel_initializer=kernel_init_actor)(self._actor)
        self._actor = Dense(units=self.action_space_size, activation=policy)(self._actor)

        # critic network branch
        self._critic = Dense(units=hidden_layers_critic[0],
                             activation=hidden_act_critic,
                             kernel_initializer=kernel_init_critic)(input)

        for i, neurons in enumerate(hidden_layers_critic[1:]):
            self._critic = Dense(units=neurons,
                                 activation=hidden_act_critic,
                                 kernel_initializer=kernel_init_critic)(self._critic)
        self._critic = Dense(units=self.action_space_size, )(self._critic)

        # build model (parent class)
        super(ActorCriticModel, self).__init__(inputs=input, outputs=[self._actor, self._critic])

class LegacyActorCriticAgentModel(Model):
    """
    Subclassed keras.Model to control the passes through the network and customize the layers with more fine control
    """
    def __init__(self, action_space_size, state_space_size,
                 hidden_layers_actor=(258, 64), hidden_act_actor='relu', kernel_init_actor='he_uniform',
                 hidden_layers_critic=(258, 64), hidden_act_critic='relu', kernel_init_critic='he_uniform',
                 policy="relu"):
        """

        """
        super(LegacyActorCriticAgentModel, self).__init__()

        self.action_space_size = action_space_size
        self.state_space_size = state_space_size

        # actor network branch
        self._actor_stack = np.empty(len(hidden_layers_actor) + 1, dtype=object)
        self._actor_stack[0] = Dense(units=hidden_layers_actor[0], input_dim=self.state_space_size,
                                          activation=hidden_act_actor, kernel_initializer=kernel_init_actor)

        for i, neurons in enumerate(hidden_layers_actor[1:]):
            self._actor_stack[i + 1] = Dense(units=neurons, input_dim=hidden_layers_actor[i],
                                              activation=hidden_act_actor, kernel_initializer=kernel_init_actor)

        self._actor_stack[-1] = Dense(units=self.action_space_size,)


        # critic network branch
        self._critic_stack = np.empty(len(hidden_layers_critic) + 1, dtype=object)
        self._critic_stack[0] = Dense(units=hidden_layers_critic[0], input_dim=self.state_space_size,
                                          activation=hidden_act_critic, kernel_initializer=kernel_init_critic)

        for i, neurons in enumerate(hidden_layers_critic[1:]):
            self._critic_stack[i + 1] = Dense(units=neurons, input_dim=hidden_layers_critic[i],
                                              activation=hidden_act_critic, kernel_initializer=kernel_init_critic)

        self._critic_stack[-1] = Dense(units=1,)


    def call(self, x, **kwargs):
        value = x
        action_probability = x

        # flow through critic
        for layer in self._critic_stack:
            value = layer(value)

        # flow through actor
        for layer in self._actor_stack:
            action_probability = layer(action_probability)

        return action_probability, value