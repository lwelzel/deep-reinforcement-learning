#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import gym

def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp'.
        Borrowed from Helper.py from previous Assignment.
    '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax 
    return np.exp(z)/np.sum(np.exp(z)) # compute softmax

def argmax(x):
    ''' Own variant of np.argmax with random tie breaking.
        Borrowed from Helper.py from previous Assignment.
    '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)


class DeepQAgent:
    def __init__(self, n_inputs, action_space, learning_rate = 0.1, gamma=0.8, use_tn = False, use_er = False):
        self.n_inputs = n_inputs #can we pull this from somewhere else?
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.use_tn = use_tn
        self.use_er = use_er

        self.DeepQ_Network = self._initialize_dqn()
        if use_tn:
            self.Target_Network = self._initialize_dqn()
    
    def _initialize_dqn(self): #add params?
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.n_inputs,)))
        model.add(layers.Dense(3, activation='relu'))
        model.add(layers.Dense(self.action_space.n))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.001))
        print(model.get_weights())
        return model
    

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        s = np.reshape(s,[1,4])
        actions = self.DeepQ_Network.predict(s)
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            if np.random.uniform(0,1) < epsilon:
                a = np.random.choice(self.action_space.n)
            else:
                a = argmax(actions)
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            p = softmax(actions,temp)
            a = np.random.choice([0,1], p=p)

        return a, actions[0]

    def one_run_update(self,states,actions,rewards,Q_values):
        #compute bellmann update target
        G = rewards + self.gamma*np.max(Q_values,axis=1)
        states = np.reshape(states,[len(states),4])

        for i in range(0,len(states)):
            Q_values[i,actions[i]] = (1-self.learning_rate) * Q_values[i,actions[i]] + self.learning_rate G[i] #update taken actions
 

        self.DeepQ_Network.fit(states,Q_values,epochs=1,verbose=True)     



    
  
def learn_dqn():
    epsilon = 0.1
    batch_size = 30
    num_iterations = 1000

    env = gym.make('CartPole-v1')
    pi = DeepQAgent(4, env.action_space)

    for iter in range(num_iterations):
        s = env.reset()
        done = False

        states,actions,rewards,Q_values = [],[],[],[]

        #one training iteration
        while not done:
            a, Q_sa = pi.select_action(s, epsilon=epsilon)
            s_next,r,done,_ = env.step(a)
            env.render()
            time.sleep(0.07)   

            states.append(s)
            actions.append(a)
            rewards.append(r)
            Q_values.append(Q_sa)
  
            s = s_next
    
        print("Iteration {0}: Timesteps survived: {1}".format(iter,len(states)))

        idxs = np.random.choice(np.arange(len(states)),size=batch_size) #randomize to break temporal correlation?
        pi.one_run_update(np.array(states)[idxs],np.array(actions)[idxs],np.array(rewards)[idxs],np.array(Q_values)[idxs])

    env.close()


    
def random_move(env,render=False):
    s = env.reset()
    rewards = []
    states = []
 
    done = False
    if render:
        env.render()
  
    while not done:
        a = np.random.choice(env.action_space.n)
        s,r,done,_ = env.step(a)
        rewards.append(r)
        states.append(s)
        if render:
            env.render()
        time.sleep(0.07)   
  
    return states,rewards,actions


def main():
    do_random = False

    if do_random:
        random_move()
    else:
        learn_dqn()


if __name__ == '__main__':
    main()

