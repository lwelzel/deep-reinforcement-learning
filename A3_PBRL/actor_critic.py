#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Actor Critic Class
#
#######################

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from base_agent import BaseAgent
from actor_critic_model import ActorCriticModel

class ActorCriticAgent(object):
    def __init__(self, state_space, action_space,
                 max_reward=500,
                 exp_policy='egreedy', epsilon=1., temperature=1.,
                 anneal_method='exponential',
                 decay=0.999, epsilon_min=0.01, temp_min=0.1,
                 learning_rate=0.01, discount=0.8,
                 hidden_layers=(64, 64), hidden_act='relu', kernel_init=None,
                 name="", id=0,
                 **kwargs):
        super(ActorCriticAgent, self).__init__()

        # states and actions
        self.state_space = state_space.shape
        self.action_space = action_space
        self.n_actions = action_space.n

        # RNG
        self.rng = np.random.default_rng()

        self.model = ActorCriticModel(self.n_actions, self.state_space[0],
                 hidden_layers_actor=(258, 64), hidden_act_actor='relu', kernel_init_actor='he_uniform',
                 hidden_layers_critic=(258, 64), hidden_act_critic='relu', kernel_init_critic='he_uniform',
                 policy="softmax")

        # TODO: compile model?
        # self.model.compile(optimizer=RMSprop(lr=learning_rate), loss=[actor loss function, critic loss function])

    def select_action(self, action_probability):
        return self.rng.choice(self.n_actions, None, p=action_probability)

    def discount_rewards(self, rewards):

        # TODO: https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning
        _rewards = np.flip(rewards)

        _discount = 00000

        
    def update(self, trace_array, episode_len):
        with tf.GradientTape() as tape:
            trace_probabilities, trace_values = self.model(tf.convert_to_tensor(trace_array,
                                                                               dtype=tf.float32))
            discounted_values = 0






def custom_train_actor_critic():
    pass


if __name__ == '__main__':
    pass