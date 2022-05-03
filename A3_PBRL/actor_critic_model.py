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
                 hidden_layers_actor=(32, 16), hidden_act_actor='relu', kernel_init_actor='glorot_uniform',
                 hidden_layers_critic=(32, 16), hidden_act_critic='relu', kernel_init_critic='glorot_uniform',
                 actor_output_activation=None, critic_output_activation=None, **kwargs):
        """
        We use 'softmax' as the activation function for the actor output layer
        so that it can receive non-normalized (+/- inf) input from prior layers
        """

        self.action_space_size = action_space_size
        self.state_space_size = state_space_size


        input = Input(shape=self.state_space_size,
                      name='common_input')

        # actor network branch
        self._actor = Dense(units=hidden_layers_actor[0],
                            activation=hidden_act_actor,
                            kernel_initializer=kernel_init_actor,
                            name="actor_branch_input")(input)

        for i, neurons in enumerate(hidden_layers_actor[1:]):
            self._actor = Dense(units=neurons,
                                activation=hidden_act_actor,
                                kernel_initializer=kernel_init_actor,
                                name=f"actor_layer_{i+1}")(self._actor)
        self._actor = Dense(units=self.action_space_size, activation=actor_output_activation,
                            name='action')(self._actor)

        # critic network branch
        self._critic = Dense(units=hidden_layers_critic[0],
                             activation=hidden_act_critic,
                             kernel_initializer=kernel_init_critic,
                             name="critic_branch_input")(input)

        for i, neurons in enumerate(hidden_layers_critic[1:]):
            self._critic = Dense(units=neurons,
                                 activation=hidden_act_critic,
                                 kernel_initializer=kernel_init_critic,
                                 name=f"critic_layer_{i+1}")(self._critic)
        self._critic = Dense(units=1,
                             name='value', activation=critic_output_activation)(self._critic)

        # build model (parent class)
        super(ActorCriticModel, self).__init__(inputs=input, outputs=[self._actor, self._critic])

class LegacyActorCriticAgentModel(Model):
    """
    Subclassed keras.Model to control the passes through the network and customize the layers with more fine control
    """
    def __init__(self, action_space_size, state_space_size,
                 hidden_layers_actor=(258, 64), hidden_act_actor='relu', kernel_init_actor='he_uniform',
                 hidden_layers_critic=(258, 64), hidden_act_critic='relu', kernel_init_critic='he_uniform',
                 actor_prop_rescale_policy="relu"):
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




if __name__ == '__main__':
    from tensorflow.keras.utils import plot_model
    a2c_model = ActorCriticModel(action_space_size=2, state_space_size=4,
                                 hidden_layers_actor=(258, 64),
                                 hidden_layers_critic=(258, 64))
    s = [0.3, 1.5, -3., 4.2]
    prob, val = a2c_model(np.reshape(s, (1, 4)))
    print(prob, val)
    a2c_model.summary()
    plot_model(a2c_model, "example_a2c_graph.png", show_shapes=True)
