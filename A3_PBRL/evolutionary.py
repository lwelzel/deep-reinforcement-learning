#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Evolutionary Agent
#
#######################

import tensorflow as tf
import numpy as np
from base_agent import BaseAgent


class EvolutionaryAgent(BaseAgent):
    def __init__(self, state_space, action_space,
                 num_agents=25, elite_percentage=0.1, initial_mean=0, initial_std=1,
                 fit_method='simple',
                 **kwargs):
        super().__init__(state_space, action_space, name='evolutionary', **kwargs)

        self.num_agents = num_agents
        self.u = elite_percentage
        self.elite_size = int(np.ceil(self.num_agents * self.u))

        self.network = None
        self.kernel_init = tf.random_normal_initializer
        self.mean = initial_mean
        self.std = initial_std
        self.fit_method = fit_method

        self.agents = []
        self.agent_returns = np.zeros(num_agents, dtype=float)
        self.create_agents()

    def create_agents(self):
        """Creates a generation of agents"""
        self.agents = []  # delete all existing agents
        for i in range(self.num_agents):
            agent = self._create_neural_net(hidden_layers=self.hidden_layers,
                                            hidden_act=self.hidden_act,
                                            kernel_init=self.kernel_init(mean=self.mean, stddev=self.std),
                                            optimizer=self.optimizer,
                                            verbose=False)
            self.agents.append(agent)

    def update_policy(self):
        """Evolves the agents one generation after having received returns"""
        elite_weights = self.select_elite(self.agent_returns)
        self.mean, self.std = self.fit_gauss(elite_weights)

        self.create_agents()

        return 0

    def set_agent(self, i):
        self.network = self.agents[i]

    def collect_return(self, i, r):
        self.agent_returns[i] = r

    def select_elite(self, rewards):
        """Given a set of full rewards, gives """
        elite_idxs = np.argpartition(rewards, -self.elite_size)[-self.elite_size:]

        elite_weights = []
        for idx in elite_idxs:
            elite_weights.append(self.agents[idx].get_weights())

        return elite_weights

    def fit_gauss(self, weights):
        """Fit a gaussian to the weights of the elite set"""
        if self.fit_method == 'simple':
            # Cast all weights in one flat array and compute the complete mean and standard deviation
            flat_weights = np.array([])
            for weight in weights:
                flat = np.concatenate([sublist.flatten() for sublist in weight])
            flat_weights = np.concatenate([flat_weights, flat])

            return np.nanmean(flat_weights), np.nanstd(flat_weights)
