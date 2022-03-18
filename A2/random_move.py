#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gym

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

    env.close()
    print("Ran for {} timesteps".format(len(rewards)))


if __name__ == '__main__':
    random_move()