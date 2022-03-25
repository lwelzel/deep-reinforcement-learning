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
    def __init__(self, env):
        self.DeepQ_Network = _initialize_dqn()
        self.Target_Network = _initialize_dqn()
    
    def _initialize_dqn(): #add params?
        model = keras.Sequential()
        model.add(keras.Input(shape=(4,)))
        model.add(layers.Dense(3, activation='relu'))
        model.add(layers.Dense(2))
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.001))
        return model
    
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        actions = self.DeepQ_Network.predict(s)
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            if np.random.uniform(0,1) < epsilon:
                a = np.random.randint(0,1)
            else:
                a = argmax(actions)
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            p = softmax(actions,temp)
            a = np.random.choice([0,1], p=p)
            
        return a
    
    def take_action(a):
        #env.step, outside class?
        






def random_move():
    env = gym.make('CartPole-v1')
    s = env.reset()
    rewards = []
    
    done = False
    env.render()
    
    while not done:
        a = np.random.choice(env.action_space.n)
        s,r,done,_ = env.step(a)
        rewards.append(r)
        env.render()
        time.sleep(0.07)
    
    env.close()
    print('Ran for {} timesteps'.format(len(rewards)))



def main():
    do_random = True

    if do_random:
        random_move()


if __name__ == '__main__':
    main()

