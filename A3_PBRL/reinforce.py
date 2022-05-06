#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Reinforce Class
#
#######################

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym
from base_agent import BaseAgent


class ReinforceAgent(BaseAgent):
    def __init__(self, state_space, action_space, **kwargs):
        super().__init__(state_space, action_space, name='reinforce', **kwargs)

    def update_policy(self, trace_array, episode_len):
        """REINFORCE weights update through automatic differentiation of a loss function"""
        num_traces = trace_array.shape[0]
        loss = 0

        with tf.GradientTape() as tape:  # Tensorflow handles differentiation
            for i, trace in enumerate(trace_array):
                cumu_trace_reward = 0
                num_steps = episode_len[i]
                trace = trace[:4 * num_steps]
                for t in reversed(range(num_steps)):
                    s = trace[4 * t]
                    a, r = int(trace[(4 * t) + 1][0]), int(trace[(4 * t) + 2][0])

                    cumu_trace_reward = r + self.discount * cumu_trace_reward

                    s_tensor = tf.constant(s, shape=(1, 4))

                    log_prob = self.get_log_prob_tf(s_tensor, a)
                    loss += (log_prob * cumu_trace_reward)
            loss = -loss / num_traces

        loss_grad = tape.gradient(loss, self.network.trainable_variables)
        self.grad_descent(loss_grad)

        return loss

    def get_log_prob_tf(self, s, a):
        """Get action probabilities (normalized to 1) using tf.tensors"""
        actions = self.network(s)[0]
        dist = tfp.distributions.Categorical(logits=actions)
        return dist.log_prob(a)



        ###########################################################################################
    #def update_policy(self, trace_array, episode_len):
    #    """"Reinforce weights update function as described by Algorithm ??"""
    #    weight_grad = np.zeros_like(self.network.get_weights(), dtype=object)
    #
    #    loss = 0
    #    num_traces = trace_array.shape[0]
    #    for i, trace in enumerate(trace_array):
    #        cumu_trace_reward = 0
    #        num_steps = episode_len[i]
    #        trace = trace[:4 * num_steps]  # remove all trailing zeros
    #        for t in reversed(range(num_steps)):
    #            s = trace[4 * t]
    #            a, r = int(trace[(4 * t) + 1][0]), int(trace[(4 * t) + 2][0])
    #
    #            cumu_trace_reward = r + self.discount * cumu_trace_reward
    #            grad_log, log_prob = self.one_step_weight_grad(s, a, cumu_trace_reward, num_traces)
    #
    #            weight_grad += grad_log
    #            loss += (log_prob * cumu_trace_reward)
    #    loss = -loss/num_traces
    #
    #    self.update_weights(weight_grad)
    #
    #    return loss
    #
    #
    #def one_step_weight_grad(self, s, a, cumu_trace_reward, num_traces):
    #    """Computes the one step gradient update"""
    #    s_tensor = tf.constant(s, shape=(1, 4))
    #    with tf.GradientTape() as tape:
    #        log_prob = self.get_log_prob_tf(s_tensor, a)
    #
    #    grad_log = tape.gradient(log_prob, self.network.trainable_variables)
    #    grad_log = [(cumu_trace_reward/num_traces) * item.numpy() for item in grad_log]
    #
    #    return grad_log, log_prob
    #
    ############################################################




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
