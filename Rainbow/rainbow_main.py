import os
import sys
import gym
import retro
import random
import argparse
import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F

from itertools import count
from retro_wrappers import *
from model import QNet
from memory import Memory

from config import *


def get_action(state, target_net, epsilon, env):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)


def update_target_model(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())

def train(render):
    online_net = QNet(h=84, w=84, outputs=36)
    online_net.load_state_dict(torch.load('saved/online_net.pt'))
    target_net = QNet(h=84, w=84, outputs=36)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    memory = torch.load('saved/model_memory.pt')
    epsilon = 0.1
    steps = 0
    beta = beta_start
    loss = 0

    for e in range(100000):
        #level = random.choice(LEVEL_SET)
        level = 'Level01'
        env = make_retro(game=env_name, state=level, use_restricted_actions=retro.Actions.DISCRETE)
        
        done = False

        total_reward = 0.0
        state = env.reset()
        state = torch.Tensor(state).to(device).permute(2,0,1)
        #state = state.view(state.size()[0], -1)
        state = state.unsqueeze(0)

        while not done:
            steps += 1
            action = get_action(state.to(device), target_net, epsilon, env)

            if render:
                env.render()
                
            next_state, reward, done, info = env.step(action)

            next_state = torch.Tensor(next_state).permute(2,0,1)
            #next_state = next_state.view(next_state.size()[0], -1)
            next_state = next_state.unsqueeze(0)
            
            total_reward += reward
            
            mask = 0 if done else 1
            action_one_hot = torch.zeros(36)
            action_one_hot[action] = 1

            reward = torch.tensor([info['score']]).to(device)            
            memory.push(state, next_state, action_one_hot, reward, mask)

            state = next_state

            if len(memory) > initial_exploration:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.02)
                beta += 0.00005
                beta = min(1, beta)

                batch, weights = memory.sample(batch_size, online_net, target_net, beta)
                loss = QNet.train_model(online_net, target_net, optimizer, batch, weights)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        if e % 1 == 0:
            print('{} episode | Total Reward: {}'.format(
                e, total_reward))
            torch.save(online_net.state_dict(), 'saved/online_net.pt')
            torch.save(memory, 'saved/model_memory.pt')
        env.close()
        
def test(level_list, render=True):
    online_net = QNet(h=84, w=84, outputs=36)
    online_net.load_state_dict(torch.load('saved/online_net.pt'))

    online_net.to(device)
    
    cnt = 0
    death = 0
    total_reward = 0.0

    str_level_list = [LEVEL_SET[idx-1] for idx in level_list]
    for level in str_level_list:
        env = make_retro(game=env_name, state=level, use_restricted_actions=retro.Actions.DISCRETE)
        
        obs = env.reset()
        state = torch.Tensor(obs).to(device).permute(2,0,1)
        #state = state.view(state.size()[0], -1)
        state = state.unsqueeze(0)

        previous_lives = 3
        previous_level = level_list[cnt] 
        cnt += 1
        if death >= 3:
            break
            
        for t in count():
            action = online_net.get_action(state.to(device))

            if render:
                env.render()
                time.sleep(0.02)

            next_state, reward, done, info = env.step(action)    
            
            next_state = torch.Tensor(next_state).permute(2,0,1)
            #next_state = next_state.view(next_state.size()[0], -1)
            next_state = next_state.unsqueeze(0)
            
            total_reward += reward
            
            current_lives = info['lives']
            current_level = info['level']
            
            if current_lives != previous_lives:
                print('Dead')
                previous_lives = info['lives']
                death += 1
                #if death >= 3:
                #    print("Finished ", level, " Total reward: {}".format(total_reward))
                #    break

            if current_level != previous_level:
                print('Stage changed')
                print("Finished ", level, " Total reward: {}".format(total_reward))
                break

            state = next_state
            
            if done:
                print('All lives gone')
                print("Finished ", level, " Total reward: {}".format(total_reward))
                break
                
        env.close()
    return

            
            
if __name__=="__main__":
    
    #train(render=False)
    test([1,2,3,4], render=True)
