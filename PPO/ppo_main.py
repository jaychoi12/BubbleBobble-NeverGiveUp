import os
import sys
import gym
import retro
import random
import time
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import PPO 
from tensorboardX import SummaryWriter

from itertools import count
from retro_wrappers import *
from memory import Memory
from config import *

def train(render=True):
    net = PPO(num_inputs, num_actions)
    #net.load_state_dict(torch.load('saved/net.pt'))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    net.to(device)
    net.train()
    running_score = 0
    steps = 0
    loss = 0

    for e in range(100000):
        #level = random.choice(LEVEL_SET)
        level = 'Level01'
        env = make_retro(game=env_name, state=level, use_restricted_actions=retro.Actions.DISCRETE)
        
        done = False
        memory = Memory()

        total_reward = 0.0
        state = env.reset()
        state = torch.Tensor(state).to(device).permute(2,0,1).view(1, -1)
        #state = state.unsqueeze(0)

        while not done:
            steps += 1
            action = net.get_action(state.to(device))
            
            if render:
                env.render()
            
            next_state, reward, done, info = env.step(action)

            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            total_reward += reward
            
            mask = 0 if done else 1
            
            action_one_hot = torch.zeros(36)
            action_one_hot[action] = 1
            
            reward = torch.tensor([info['score']]).to(device)
            memory.push(state, next_state, action_one_hot, reward, mask)

            state = next_state.view(1,-1)

        loss = PPO.train_model(net, memory.sample(), optimizer)

        if e % log_interval == 0:
            print('{} episode | Total Reward: {}'.format(
                e, total_reward))
            torch.save(net.state_dict(), 'saved/net.pt')
            
        env.close()
        
def test(level_list, render=True):
    net = PPO(num_inputs, num_actions)
    net.load_state_dict(torch.load('saved/net.pt'))

    net.to(device)
    
    cnt = 0
    death = 0
    str_level_list = [LEVEL_SET[idx-1] for idx in level_list]
    for level in str_level_list:
        env = make_retro(game=env_name, state=level, use_restricted_actions=retro.Actions.DISCRETE)
        
        obs = env.reset()
        state = torch.Tensor(obs).to(device).permute(2,0,1).view(1, -1)
        total_reward = 0.0

        previous_lives = 3
        previous_level = level_list[cnt] 
        cnt += 1
        if death >= 3:
            break
            
        for t in count():
            action = net.get_action(state.to(device))

            if render:
                env.render()
                time.sleep(0.02)

            next_state, reward, done, info = env.step(action)    
            
            total_reward += reward
            
            current_lives = info['lives']
            current_level = info['level']
            
            if current_lives != previous_lives:
                print('Dead')
                previous_lives = info['lives']
                death += 1
                if death >= 3:
                    print("Finished ", level, " Total reward: {}".format(total_reward))
                    break

            if current_level != previous_level:
                print('Stage changed')
                print("Finished ", level, " Total reward: {}".format(total_reward))
                break

            state = torch.Tensor(next_state).to(device).permute(2,0,1).view(1, -1)

            if done:
                print('All lives gone')
                print("Finished ", level, " Total reward: {}".format(total_reward))
                break
                
        env.close()
    return


if __name__=="__main__":
    env = make_retro(game=env_name, use_restricted_actions=retro.Actions.DISCRETE)
    
    num_inputs = env.observation_space.shape[0] * env.observation_space.shape[1]
    num_actions = env.action_space.n
    env.close()

    train(render=False)
    #test([8,3,2,3], render=True)
