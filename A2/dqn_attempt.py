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
        model.compile() #loss='mean_squared_error', optimizer=optimizers.Adam(0.001))
        print(model.get_weights())
        return model
    

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        actions = self.DeepQ_Network(s)#(s[0],s[1],s[2],s[3]))
        print(actions)
        
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
            
        return a
    
    def take_action(a):
        pass;
        #env.step, outside class?
        

def learn_dqn():
    epsilon = 0.1

    env = gym.make('CartPole-v1')
    pi = DeepQAgent(4, env.action_space)

    s = env.reset()
    print(s)
    print(s.shape[-1])
    done = False
    while not done:
        a = pi.select_action(s, epsilon=epsilon)
        s_next,r,done,_ = env.step(a)
        env.render()
        s = s_next
        


    
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

