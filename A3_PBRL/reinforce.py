#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Reinforce Agent Class
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
                num_steps = episode_len[i]
                trace = trace[:4 * num_steps]
                discounted_rewards = np.zeros(num_steps)
                # Compute the discounted_rewards backwards to easily account for future discount
                for t in reversed(range(num_steps)):
                    r = int(trace[(4 * t) + 2][0])
                    discounted_rewards[t] = r + self.discount * discounted_rewards[np.clip(t+1, 0, num_steps-1)]

                # Normalize discounted rewards
                discounted_rewards = (discounted_rewards - np.mean(discounted_rewards))/(np.std(discounted_rewards) + 1e-9)

                for t in range(num_steps):
                    s = trace[4 * t]
                    a = int(trace[(4 * t) + 1][0])

                    s_tensor = tf.constant(s, shape=(1, 4))

                    log_prob = self.get_log_prob_tf(s_tensor, a)
                    loss += (log_prob * discounted_rewards[t])
            loss = -loss / num_traces

        loss_grad = tape.gradient(loss, self.network.trainable_variables)
        self.grad_descent(loss_grad)

        return loss

    def get_log_prob_tf(self, s, a):
        """Get action probabilities (normalized to 1) using tf.tensors"""
        actions = self.network(s)[0]
        dist = tfp.distributions.Categorical(logits=actions) # logits because actions isn't normalized
        return dist.log_prob(a)

    
    def grad_descent(self, weights_grad):
        """Update the network weights using gradient from loss function"""
        weights = self.network.get_weights()
        new_weights = np.array([weight - (weight_grad * self.learning_rate)
                                for weight, weight_grad
                                in zip(weights, weights_grad)],
                               dtype=object)

        self.network.set_weights(new_weights)


 


def main():
    env = gym.make('CartPole-v1')
    pi = ReinforceAgent(env.observation_space, env.action_space)


if __name__ == '__main__':
    main()
