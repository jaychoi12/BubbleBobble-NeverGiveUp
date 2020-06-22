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
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(36)]], dtype=torch.int64).to(device)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    batch, weights = memory.sample(BATCH_SIZE, GAMMA, policy_net, target_net, BETA, ALPHA)
    
    td_error = get_td_error(BATCH_SIZE, GAMMA, policy_net, target_net, batch.state, batch.action, batch.next_state, batch.reward)
    loss = pow(td_error, 2) * weights
    loss = loss.mean()
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    

def train(n_episodes, render=False):
    policy = torch.load('saved/model.pt')
    for episode in range(n_episodes):
        state = random.choice(STATE_SET)
        env = make_retro(game=GAME, state=state, use_restricted_actions=retro.Actions.DISCRETE)
        
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state)
            if render:
                env.render()
                
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            
            reward = torch.tensor([reward]).to(device)
            
            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state
            
            if steps_done > INITIAL_MEMORY:
                optimize_model()
                
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    
            if done:
                break
                
        if episode % 20 == 0:
            print('Total steps: {} \t Episode: {}\{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
            torch.save(policy_net, 'saved/model.pt')
        env.close()
    return
            
def test(n_episodes, policy, render=True):
    policy = torch.load('saved/model.pt')
    for episode in range(n_episodes):
        state = random.choice(STATE_SET)
        env = make_retro(game=GAME, state=state, use_restricted_actions=retro.Actions.DISCRETE)
        
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(device)).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return
    
if __name__ == '__main__':
        
    #create network
    policy_net = Duel_DQN(h=84, w=84, outputs=36).to(device)
    target_net = Duel_DQN(h=84, w=84, outputs=36).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup get
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0
    
    #initialize replay memory
    memory = PER_Memory(MEMORY_SIZE, SMALL_EPSILON) 
    
    #train_model
    #train(1000000, render=False)
    test(1, policy_net, render=True)
    