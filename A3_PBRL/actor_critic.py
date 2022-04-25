#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Actor Critic Class
#
#######################

from base_agent import BaseAgent

class ActorCriticAgent(BaseAgent):
    def __init__(self, **kwargs):
        def __init__(self, state_space, action_space, **kwargs):
            super().__init__(state_space, action_space, **kwargs)
        
    def update(self, trace_array, episode_len):
        print(f"Average Reward: {np.mean(episode_len)}")



def custom_train_actor_critic():
    pass


if __name__ == '__main__':
    main()
