#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Evolutionary Agent
#
#######################

import tensorflow as tf
import numpy as np
import time
from base_agent import BaseAgent


class EvolutionaryAgent(BaseAgent):
    def __init__(self, state_space, action_space,
                 num_agents=25, elite_percentage=0.1, initial_mean=0, initial_std=1,
                 kernel_init=tf.random_normal_initializer, fit_method='simple',
                 **kwargs):
        super().__init__(state_space, action_space, name='evolutionary', kernel_init=kernel_init, **kwargs)

        self.num_agents = num_agents
        self.u = elite_percentage
        self.elite_size = int(np.ceil(self.num_agents * self.u))

        self.initial_mean = initial_mean
        self.initial_std = initial_std
        self.fit_method = fit_method

        if self.fit_method == 'simple':
            # Single mean and stddev for all weights
            self.mean = initial_mean
            self.std = initial_std
        elif self.fit_method == 'individual':
            # Each weight has own mean and stddev
            self.mean = np.zeros_like(self.network.get_weights(), dtype=object)
            self.std = np.zeros_like(self.network.get_weights(), dtype=object)

        else:
            raise ValueError(f"Unknown fit method {self.fit_method}")

        # Delete network created by BaseAgent, create new list of agents
        self.network = None
        self.agents = []
        self.agent_returns = np.zeros(num_agents, dtype=float)
        self.initialize_agents()

    def initialize_agents(self):
        """Creates a generation of agents"""
        self.agents = []  # delete all existing agents
        for i in range(self.num_agents):
            agent = self._create_neural_net(hidden_layers=self.hidden_layers,
                                            hidden_act=self.hidden_act,
                                            kernel_init=self.kernel_init(mean=self.initial_mean, stddev=self.initial_std),
                                            optimizer=self.optimizer,
                                            verbose=False)
            self.agents.append(agent)

    def update_policy(self):
        """Evolves the agents one generation after having received returns"""
        t0 = time.time()
        elite_weights = self.select_elite(self.agent_returns)
        self.fit_gauss(elite_weights)

        self.evolve_agents()
        print(f"Updating Generation took {time.time()-t0} seconds")
        return 0  # training func. excepts a loss value

    def evolve_agents(self):
        """"""
        if self.fit_method == 'simple':
            self.initialize_agents()
        elif self.fit_method == 'individual':
            for agent in self.agents:
                new_weights = np.zeros_like(agent.get_weights())
                for i in range(len(agent.get_weights())):
                    new_weights[i] = np.random.normal(self.mean[i], self.std[i], self.mean[i].shape)
                    if i == 5:
                        print(new_weights[i])
                agent.set_weights(new_weights)

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

            self.mean = np.nanmean(flat_weights)  # Ignore NaN so that network can fix itself if a few agents go bad
            self.std = np.nanstd(flat_weights)

        elif self.fit_method == 'individual':
            for set in range(len(weights[0])):
                w_set = []
                for i in range(self.elite_size):
                    w_set.append(weights[i][set])
                self.mean[set] = np.nanmean(w_set, axis=0)
                self.std[set] = np.nanstd(w_set, axis=0)

