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
        super().__init__(state_space, action_space, **kwargs)

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
        dist = tfp.distributions.Categorical(logits=actions) # TODO: tf.random.Categorical ?
        return dist.log_prob(a)


 


def main():
    pass


if __name__ == '__main__':
    main()
