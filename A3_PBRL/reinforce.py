#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Reinforce Class
#
#######################

import numpy as np
import tensorflow as tf
import gym
from base_agent import BaseAgent


class ReinforceAgent(BaseAgent):
    def __init__(self, state_space, action_space, **kwargs):
        super().__init__(state_space, action_space, name='reinforce', **kwargs)

    def update_policy(self, trace_array, episode_len, train_length):
        """"Reinforce weights update function as described by Algorithm """
        weight_grad = np.zeros_like(self.network.get_weights(), dtype=object)

        num_traces = trace_array.shape[0]
        for i, trace in enumerate(trace_array):
            cumu_trace_reward = 0
            num_steps = episode_len[i]
            trace = trace[:4 * episode_len[i]]  # remove all trailing zeros
            for t in reversed(range(num_steps)):
                s = trace[4 * t]
                a, r = int(trace[(4 * t) + 1][0]), int(trace[(4 * t) + 2][0])

                cumu_trace_reward = r + self.discount * cumu_trace_reward
                weight_grad += self.one_step_weight_grad(s, a, cumu_trace_reward, train_length)


        self.update_weights(weight_grad)

    def one_step_weight_grad(self, s, a, cumu_trace_reward, train_length):
        """Computes the one step gradient update"""
        x = tf.constant(s, shape=(1, 4))
        with tf.GradientTape() as tape:
            val_sa = self.network(x)[0][a]
            log_val_sa = tf.math.log(val_sa)
        grad_log = tape.gradient(log_val_sa, self.network.trainable_variables)
        grad_log = [(cumu_trace_reward/train_length) * item.numpy() for item in grad_log]

        return grad_log


    # def loss_function(self, trace_array, episode_len):
    #    """Full loss function to push through automatic differentiation"""
    #    num_traces = trace_array.shape[0]
    #    for i, trace in enumerate(trace_array)


def custom_train_reinforce():
    """Function to train a REINFORCE agent on the cartpole-v1 environment for testing"""

    ##### PARAMETERS #####
    num_traces = 1
    ######################

    env = gym.make('CartPole-v1')
    pi = ReinforceAgent(env.observation_space, env.action_space)


def main():
    custom_train_reinforce()


if __name__ == '__main__':
    main()
