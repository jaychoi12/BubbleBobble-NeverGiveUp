import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time

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


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2,0,1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def select_action(state):
    global steps_done
    steps_done += 1
    return torch.tensor([[random.randrange(36)]], dtype=torch.int64).to(device)

def optimize_model(memory, embedding, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    batch = memory.sample(BATCH_SIZE)

    state_batch = torch.cat(batch.state).to(device)
    actions = tuple((map(lambda a: torch.tensor([[a]]), batch.action)))
    action_batch = torch.cat(actions).to(device)
    next_state_batch = torch.cat(batch.next_state).to(device)    
    
    feature1 = embedding(state_batch)
    feature2 = embedding(next_state_batch)
    
    loss = embedding.compute_loss(feature1, feature2, action_batch)
    
    optimizer.zero_grad()
    loss.backward()
    for param in embedding.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss
    

def train(n_episodes):
    embedding = ICM(h=84, w=84, outputs=36).to(device)
    embedding.load_state_dict(torch.load('saved/embedding.pt'))
    
    optimizer = optim.Adam(embedding.parameters(), lr=pretrain_lr)
    
    memory = ReplayMemory(MEMORY_SIZE, Transition) 
    memory = torch.load('saved/ICM_memory.pt')
    
    for episode in range(n_episodes):
        level = random.choice(LEVEL_SET)
        #level = 'Level01'
        env = pretrain_make_retro(game=GAME, state=level, use_restricted_actions=retro.Actions.DISCRETE)
        
        obs = env.reset()
        state = get_state(obs)
        
        total_reward = 0.0
        loss = float('inf')
        for t in count():
            action = select_action(state)
                
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            
            if not done:
                next_state = get_state(obs)
                reward = torch.tensor([reward]).to(device)
            
                memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
                state = next_state
                
                if len(memory) > INITIAL_MEMORY:
                    loss = optimize_model(memory, embedding, optimizer)
        
            else:
                break
                
        if episode % 1 == 0:
            print('Total steps: {} \t Loss: {}'.format(steps_done, loss))
            torch.save(embedding.state_dict(), 'saved/embedding.pt')
            torch.save(memory, 'saved/ICM_memory.pt')
        env.close()
    return
            

if __name__ == '__main__':
        
    steps_done = 0
    
    #train_model
    train(1000000)
    