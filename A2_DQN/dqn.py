#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import gym
from helper import softmax, argmax



class DeepQAgent:
    def __init__(self,  n_inputs, 
                        action_space, 
                        hidden_layers=None,
                        hidden_act = 'relu',
                        learning_rate = 0.01, 
                        gamma=0.8, 
                        use_tn = False, 
                        use_er = False
                        ):


        self.n_inputs = n_inputs #can we pull this from somewhere else?
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
            
        self.use_tn = use_tn
        self.use_er = use_er

        self.DeepQ_Network = self._initialize_dqn(hidden_layers,hidden_act)
        if use_tn:
            self.Target_Network = self._initialize_dqn(hidden_layers,hidden_act)
    

    def _initialize_dqn(self,hidden_layers=None,hidden_act='relu',init='HeUniform'): 
        """Template model for both Main and Target Network for the Q-value mapping. Layers should be a list with the number of fully connected nodes per hidden layer"""
    
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.n_inputs,)))

        if layers == None:
            print("WARNING: No hidden layers given for Neural Network")
            input("Continue? ... ")
        else:
            for n_nodes in hidden_layers:
                model.add(layers.Dense(n_nodes,activation=hidden_act,kernel_initializer=init))

        model.add(layers.Dense(self.action_space.n,kernel_initializer=init))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(self.learning_rate))

        return model
    

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        """Function that takes a single state as input and uses the Main or Target Network combined with some exploration strategy
           to select an action for the agent to take. It returns this action, and the Q_values computed by the Network"""

        s = np.reshape(s,[1,4])
        actions = self.DeepQ_Network.predict(s)[0]
        
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


    def one_step_update(self,s,a,r,s_next,done):
        Q_current = self.DeepQ_Network.predict(np.reshape(s,[1,4]))

        #update targets
        if done:
            G = r
        else:
            if self.use_tn:
                pass; #TODO
            else:
                max_Q_next = np.max(self.DeepQ_Network.predict(np.reshape(s_next,[1,4])))
            G = r * self.gamma * max_Q_next

        Q_current[0,a] = (1-self.learning_rate)*Q_current[0,a] + self.learning_rate * G

        self.DeepQ_Network.fit(np.reshape(s,[1,4]),Q_current,verbose=0)


    def update(self,states,actions,rewards,states_next,done_ar):
        """Receives numpy_ndarrays of size batch_size and computes new back-up targets on which the model is trained"""

        Q_current = self.DeepQ_Network.predict(states)#Current Q(s,a) values to be updated if a is taken

        for i in range(0,len(states)):
            if done_ar[i]:
                G = rewards[i] #don't bootstrap
            else:
                if self.use_tn:
                    pass; #TODO
                else:
                    next_state = np.reshape(states_next[i],[1,4])
                    max_Q_next = np.max(self.DeepQ_Network.predict(next_state))
                G = rewards[i] + self.gamma*max_Q_next
                
            #Bellmann's equation
            Q_current[i,actions[i]] = (1-self.learning_rate)*Q_current[i,actions[i]] + self.learning_rate * G         
                 

        states = np.reshape(states,[len(states),4]) #reshape to feed into keras

        #Fit and train the network
        self.DeepQ_Network.fit(states,Q_current,epochs=1,verbose=True)     



    
  
def learn_dqn():
    epsilon = 1.
    batch_size = 128
    num_iterations = 1000

    env = gym.make('CartPole-v1')
    pi = DeepQAgent(4, env.action_space, hidden_layers=[12,6],use_er = False)

    all_rewards = []
    states,actions,rewards,states_next,done_ar = [],[],[],[],[]

    for iter in range(num_iterations):
        s = env.reset()
        done = False
        episode_reward = 0

        #one training iteration
        while not done:
            a, Q_sa = pi.select_action(s, epsilon=epsilon)
            s_next,r,done,_ = env.step(a)
            env.render()
            time.sleep(1e-3)   

            episode_reward += r
            if pi.use_er:
                states.append(s)
                actions.append(a)
                rewards.append(r)
                states_next.append(s_next)
                done_ar.append(done)
  
            #'Dumb' 1-step update
            pi.one_step_update(s,a,r,s_next,done)#np.array(s),np.array(a),np.array(r),np.array(s_next),np.array(done))

            s = s_next

            

        all_rewards.append(episode_reward)
        print("Iteration {0}: Timesteps survived: {1} ({2})".format(iter,int(episode_reward),epsilon))

        #epsilon annealing schedule?
        if epsilon > 0.1 and (iter+1) % 25 == 0:
            epsilon = epsilon - 0.1#(1./num_iterations)

    #Plot learning curve
    fig = LearningCurvePlot(title="No Target Network, No Experience Replay")
    fig.add_curve(smooth(all_rewards,100))
        

        #form of experience replay
        #if len(states) > 500:
        #    idxs = np.random.choice(np.arange(len(states)),size=batch_size) #randomize to break temporal correlation?
        #    pi.update(np.array(states)[idxs],np.array(actions)[idxs],np.array(rewards)[idxs],np.array(states_next)[idxs],np.array(done_ar)[idxs])       



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

