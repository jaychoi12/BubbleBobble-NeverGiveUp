from collections import namedtuple, deque
from utils import *
from config import *
import random
import numpy as np


Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class PER_Memory(object):
    def __init__(self, capacity, small_epsilon):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.memory_probabiliy = deque(maxlen=capacity)
        self.small_epsilon = small_epsilon

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) > 0:
            max_probability = max(self.memory_probabiliy)
        else:
            max_probability = self.small_epsilon
            
        self.memory.append(Transition(*args))
        self.memory_probabiliy.append(max_probability)

    def sample(self, batch_size, gamma, policy_net, target_net, beta, alpha):
        probability_sum = sum(self.memory_probabiliy)
        p = [probability / probability_sum for probability in self.memory_probabiliy]

        indexes = np.random.choice(np.arange(len(self.memory)), batch_size, p=p)
        transitions = [self.memory[idx] for idx in indexes]
        transitions_p = [p[idx] for idx in indexes]
        batch = Transition(*zip(*transitions))

        weights = [pow(self.capacity * p_j, -beta) for p_j in transitions_p]
        weights = torch.Tensor(weights).to(device)
        weights = weights / weights.max()

        td_error = get_td_error(batch_size, gamma, policy_net, target_net, batch.state, batch.action, batch.next_state, batch.reward)

        td_error_idx = 0
        for idx in indexes:
            self.memory_probabiliy[idx] = pow(abs(td_error[td_error_idx]) + self.small_epsilon, alpha).item()
            td_error_idx += 1

        return batch, weights

    def __len__(self):
        return len(self.memory)
    