import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time
import pickle

import retro

from retro_wrappers import *
from memory import *
from models import *
from config import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

Feature = namedtuple('Feature',
                     ('embedded_feature'))

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2,0,1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def select_action(state, policy_net):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(36)]], dtype=torch.int64).to(device)

def compute_intrinsic_reward(buffer, state, embedding, prediction_net, random_net):
    #feature = embedding(state.to(device))
    #kernel = buffer.compute_kernel(feature, 10)
    #episodic_reward = 1/(torch.sqrt(kernel) + 0.001)
    
    params = list(prediction_net.parameters())
    sub_optimizer = optim.Adam(params, lr=lr)
    
    global error_list
    g_hat = prediction_net(state.to(device))
    g = random_net(state.to(device))
    error = torch.pow(torch.norm(g_hat - g, dim=-1), 2)
    error_list.append(error.item())
    
    mean = sum(error_list) / len(error_list) 
    variance = sum([((x - mean) ** 2) for x in error_list]) / len(error_list) 
    std = variance ** 0.5
    #modulator = 1 + (error - mean) / (std + 1e-6)
    error = (error - mean) / (std + 1e-6)
    
    sub_optimizer.zero_grad()
    error.backward()
    for param in prediction_net.parameters():
        if param.requires_grad == True:
            param.grad.data.clamp_(-1, 1)
    sub_optimizer.step()
    
    #intrinsic_reward = episodic_reward * min(max(modulator, 1), 5)
    intrinsic_reward = error
    return intrinsic_reward
    
def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    batch, weights = memory.sample(BATCH_SIZE, GAMMA, policy_net, target_net, BETA, ALPHA)
  
    td_error = get_td_error(BATCH_SIZE, GAMMA, policy_net, target_net, batch.state, batch.action, batch.next_state, batch.reward)
    
    loss = pow(td_error, 2) * weights.detach()
    loss = loss.mean()
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.requires_grad == True:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    
def train(n_episodes, render=False):
    policy_net = Duel_DQN(h=84, w=84, outputs=36).to(device)
    policy_net.load_state_dict(torch.load('saved/model.pt'))
    
    target_net = Duel_DQN(h=84, w=84, outputs=36).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    embedding = ICM(h=84, w=84, outputs=36).to(device)
    #embedding.load_state_dict(torch.load('saved/embedding.pt'))
    
    random_net = RND(h=84, w=84).to(device)
    random_net.load_state_dict(torch.load('saved/random_net.pt'))

    prediction_net = RND(h=84, w=84).to(device)    
    prediction_net.load_state_dict(torch.load('saved/prediction_net.pt'))
   
    memory = PER_Memory(MEMORY_SIZE, SMALL_EPSILON)
    memory = torch.load('saved/model_memory.pt')

    params = list(policy_net.parameters())
    optimizer = optim.Adam(params, lr=lr)
    
    for episode in range(n_episodes):
        buffer = ReplayMemory(BUFFER_SIZE, Feature) 
        
        #level = random.choice(LEVEL_SET)
        level = 'Level01'
        env = make_retro(game=GAME, state=level, use_restricted_actions=retro.Actions.DISCRETE)
        
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state, policy_net)
            if render:
                env.render()
                
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            
            feature = embedding(state.to(device))
            buffer.push(feature.to('cpu'))
            
            if t > INITIAL_BUFFER:
                intrinsic_reward = compute_intrinsic_reward(buffer, state, embedding, prediction_net, random_net)
                #reward = torch.tensor([reward]).to(device) #+ 0.3 * intrinsic_reward.item()
                reward = torch.tensor([info['score']]).long().to(device) + 0.3 * intrinsic_reward.item()
                memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
                if len(memory) > INITIAL_MEMORY:
                    optimize_model(memory, policy_net, target_net, optimizer)
                
                    if steps_done % TARGET_UPDATE == 0:
                        target_net.load_state_dict(policy_net.state_dict())
            
            state = next_state
            
            if done:
                break
            
        if episode % 1 == 0:
            print('Total steps: {} \t Episode: {}\{} \t Score: {} \t Total reward: {}'.format(steps_done, episode, t, info['score'], total_reward))
            torch.save(policy_net.state_dict(), 'saved/model.pt')
            #torch.save(random_net.state_dict(), 'saved/random_net.pt')
            torch.save(prediction_net.state_dict(), 'saved/prediction_net.pt')
            torch.save(memory, 'saved/model_memory.pt')
            
        env.close()
    return
            
def test(level_list, render=True):
    policy_net = Duel_DQN(h=84, w=84, outputs=36).to(device)
    policy_net.load_state_dict(torch.load('saved/model.pt'))
    
    cnt = 0
    death = 0
    total_reward = 0.0
    str_level_list = [LEVEL_SET[idx-1] for idx in level_list]
    for level in str_level_list:
        env = make_retro(game=GAME, state=level, use_restricted_actions=retro.Actions.DISCRETE)
        
        obs = env.reset()
        state = get_state(obs)

        previous_lives = 3
        previous_level = level_list[cnt] 
        cnt += 1
        if death >= 3:
            break
            
        for t in count():
            action = policy_net(state.to(device)).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)
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

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print('All lives gone')
                print("Finished ", level, " Total reward: {}".format(total_reward))
                break
                
        env.close()
    return

if __name__ == '__main__':
    
    steps_done = 0
    error_list = []

    #train_model
    #train(1000000, render=False)
    test([1,2,3,4], render=True)
