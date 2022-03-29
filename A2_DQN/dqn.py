#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import time
from time import perf_counter, strftime, gmtime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import gym
from buffer_class import MetaBuffer
from helper import LearningCurvePlot, smooth, softmax, argmax


class DeepQAgent:
    def __init__(self, n_inputs,
                 action_space,
                 learning_rate=0.01,
                 gamma=0.8,
                 hidden_layers=None,
                 hidden_act='relu',
                 init='HeUniform',
                 loss_func='mean_squared_error',
                 use_tn=False,
                 use_er=False,
                 depth=2500,
                 sample_batch_size=64
                 ):

        self.n_inputs = n_inputs  # can we pull this from somewhere else?
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.hidden_layers = hidden_layers

        self.use_tn = use_tn
        self.use_er = use_er

        self.DeepQ_Network = self._initialize_dqn(hidden_layers, hidden_act, init, loss_func)
        if use_tn:
            self.Target_Network = self._initialize_dqn(hidden_layers, hidden_act, init, loss_func)

        if use_er:
            self.buffer = MetaBuffer(depth,sample_batch_size)


    def _initialize_dqn(self, hidden_layers=None, hidden_act='relu', init='HeUniform', loss_func='mean_squared_error'):
        """Template model for both Main and Target Network for the Q-value mapping. Layers should be a list with the number of fully connected nodes per hidden layer"""

        model = keras.Sequential()
        model.add(keras.Input(shape=(self.n_inputs,)))

        if layers == None:
            print("WARNING: No hidden layers given for Neural Network")
            input("Continue? ... ")
        else:
            for n_nodes in hidden_layers:
                model.add(layers.Dense(n_nodes, activation=hidden_act, kernel_initializer=init))

        model.add(layers.Dense(self.action_space.n, kernel_initializer=init))
        model.summary()
        model.compile(loss=loss_func, optimizer=optimizers.Adam(0.001))

        return model

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        """Function that takes a single state as input and uses the Main or Target Network combined with some exploration strategy
           to select an action for the agent to take. It returns this action, and the Q_values computed by the Network"""

        # TODO: implement fast, vectorized select_action from last report
        s = np.reshape(s, [1, 4])
        if self.use_tn:
            actions = self.Target_Network.predict(s)[0]
        else:
            actions = self.DeepQ_Network.predict(s)[0]

        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            if np.random.uniform(0, 1) < epsilon:
                a = np.random.choice(self.action_space.n)
            else:
                a = argmax(actions)

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            p = softmax(actions, temp)
            a = np.random.choice([0, 1], p=p)

        return a

    def one_step_update(self, s, a, r, s_next, done):

        if self.use_tn:
            Q_current = self.Target_Network.predict(np.reshape(s, [1, 4]))
        else:
            Q_current = self.DeepQ_Network.predict(np.reshape(s, [1, 4]))

        # update targets
        if done:
            G = r
        else:
            if self.use_tn:
                max_Q_next = np.max(self.Target_Network.predict(np.reshape(s_next, [1, 4])))
            else:
                max_Q_next = np.max(self.DeepQ_Network.predict(np.reshape(s_next, [1, 4])))
            G = r * self.gamma * max_Q_next

        Q_current[0, a] = (1 - self.learning_rate) * Q_current[0, a] + self.learning_rate * G

        self.DeepQ_Network.fit(np.reshape(s, [1, 4]), Q_current, verbose=0)

    def update(self, states, actions, rewards, states_next, done_ar):
        """Receives numpy_ndarrays of size batch_size and computes new back-up targets on which the model is trained"""

        if self.use_tn:
            Q_current = self.Target_Network.predict(states)
        else:
            Q_current = self.DeepQ_Network.predict(states)  # Current Q(s,a) values to be updated if a is taken
        # TODO: compare with update from last report
        for i in range(0, len(states)):
            if done_ar[i].any():
                G = rewards[i,0]  # don't bootstrap
            else:
                next_state = np.reshape(states_next[i], [1, 4])
                if self.use_tn:
                    max_Q_next = np.max(self.DeepQ_Network.predict(next_state))
                else:
                    max_Q_next = np.max(self.DeepQ_Network.predict(next_state))
                G = rewards[i,0] + self.gamma * max_Q_next

            # Bellmann's equation
            Q_current[i, int(actions[i,0])] = (1 - self.learning_rate) * Q_current[i, int(actions[i,0])] + self.learning_rate * G

        states = np.reshape(states, [len(states), 4])  # reshape to feed into keras

        # Fit and train the network
        self.DeepQ_Network.fit(states, Q_current, epochs=1, verbose=0)

    def save(self, rewards):
        """Saves Deep Q-Network and array of rewards"""
        model_name = f"alpha{self.learning_rate}-gamma{self.gamma}-"+"".join([str(n) for n in self.hidden_layers])
        self.DeepQ_Network.save("DeepQN_{}.h5".format(model_name))
        np.savetxt("Rewards_{}.csv".format(model_name), a, delimiter=",")



def learn_dqn(learning_rate,policy,epsilon,temp,gamma,hidden_layers,use_er,use_tn,num_iterations,depth=2500,learn_freq=4,
              target_update_freq=25,sample_batch_size=128,anneal_method=None,render=False):
    """Callable DQN function for complete runs and parameter optimization"""
    env = gym.make('CartPole-v1')
    pi = DeepQAgent(4, env.action_space, learning_rate, gamma, hidden_layers, use_tn=use_tn, use_er=use_er, depth=depth, sample_batch_size=sample_batch_size)

    all_rewards = np.full(shape=num_iterations, fill_value=np.nan, dtype=np.float64) #use to keep track of learning curve
    if not anneal_method == None:
        epsilon = 1.    


    #we initialize an empty buffer, fill it in with random values first. Later additions then overwrite these random values
    if use_er:
        timesteps = 0
        while timesteps < depth+1: #+1 to make sure buffer is filled
            s = env.reset()
            done = False
            while not done:
                a = pi.select_action(s, policy, epsilon, temp)
                s_next, r, done, _ = env.step(a)
                pi.buffer.update_buffer(np.array([s,a,r,s_next,done]))
                timesteps += 1
                if render: env.render()

    for iter in range(num_iterations):
        s = env.reset()
        done = False
        episode_reward = 0

        while not done:
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done , _ = env.step(a)
            episode_reward += r
            if pi.use_er:
                pi.buffer.update_buffer(np.array([s,a,r,s_next,done]))

        
            else:
                pi.one_step_update(s,a,r,s_next,done) #one-step 'dumb' update
            
            s = s_next
            if render: env.render()

        all_rewards[iter] = episode_reward
        if render: print("Iteration {0}: Timesteps survived: {1} ({2})".format(iter, int(episode_reward),round(epsilon,2)))

        if anneal_method == 'linear': epsilon -= (1.-0.1)/(num_iterations)

        if pi.use_er:
            batch = pi.buffer.sample
            batch = np.asarray(batch).astype('float32')
            pi.update(batch[:,0],batch[:,1],batch[:,2],batch[:,3],batch[:,4])
            

        if pi.use_tn and iter % target_update_freq == 0:
            if render: print("Updating Target Network")
            pi.Target_Network.set_weights(pi.DeepQ_Network.get_weights())


    #save model and learning curve


    env.close()
    pi.save(all_rewards)

    return all_rewards

            
    
    
            
    
    
    






def play_dqn():
    """Catch all function containing the complete learning process for a DQN on the gym polecart environment to play with"""

    ###PARAMETERS###############
    epsilon = .1
    temp = 1.
    policy = 'egreedy'  # 'egreedy'

    depth = 2500
    batch_size = 128
    num_iterations = 250
    target_update_freq = 25  # iterations
    max_training_batch = int(1e6)  # storage arrays
    # training_data_shape = (max_training_batch, 1)

    e_anneal = False
    use_er = True
    use_tn = True

    render = True
    plot = True
    title = r"Softmax $\tau$=1, +TN -ER"
    ###########################

    env = gym.make('CartPole-v1')
    pi = DeepQAgent(4, env.action_space, hidden_layers=[12, 6], use_er=use_er, use_tn=use_tn, depth=depth)

    if pi.use_tn and e_anneal:
        epsilon = 0.8

    all_rewards = []
    rewards = np.full(shape=max_training_batch, fill_value=np.nan, dtype=np.float64)

    # buffer burn in time
    if use_er:
        timesteps = 0
        while timesteps < depth + 1:
            s = env.reset()
            done = False

            while not done:
                a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
                s_next, r, done, _ = env.step(a)
                if render:
                    env.render()
                    time.sleep(1e-3)
                pi.buffer.update_buffer(np.array([s, a, r, s_next, done]))
                timesteps += 1

    for iter in range(num_iterations):
        s = env.reset()
        done = False
        episode_reward = 0

        # one training iteration
        while not done:
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
            s_next, r, done, _ = env.step(a)
            if render:
                env.render()
                time.sleep(1e-3)

            episode_reward += r
            if pi.use_er:
                pi.buffer.update_buffer(np.array([s, a, r, s_next, done]))
            else:
                # 'Dumb' 1-step update
                pi.one_step_update(s, a, r, s_next, done)

            s = s_next

        all_rewards.append(episode_reward)
        rewards[iter] = episode_reward
        print("Iteration {0}: Timesteps survived: {1}".format(iter, int(episode_reward)))

        if pi.use_er:
            batch = pi.buffer.sample
            batch = np.asarray(batch).astype('float32')
            pi.update(batch[:,0],batch[:,1],batch[:,2],batch[:,3],batch[:,4])

        # epsilon annealing schedule?
        # if epsilon > 0.1 and (iter+1) % 25 == 0:
        #    epsilon = epsilon - 0.1#(1./num_iterations)

        if pi.use_tn and iter % target_update_freq == 0:
            print("Updating Target Network..")
            pi.Target_Network.set_weights(pi.DeepQ_Network.get_weights())

            # if e_anneal and iter % (4*target_update_freq) == 0 and epsilon > 1.1: #want it at ~0.1, but rounding errors can mess this up
            #    epsilon -= 0.1
            #    print("Annealed epsilon (new: {})".format(epsilon))

    # Plot learning curve
    if plot:
        fig = LearningCurvePlot(title=title)
        fig.add_curve(smooth(all_rewards, 50))

    env.close()


def main():
    play_dqn()


if __name__ == '__main__':
    main()
