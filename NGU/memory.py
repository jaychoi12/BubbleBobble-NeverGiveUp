from collections import namedtuple, deque
from utils import *
from config import *
import random
import numpy as np


Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

Feature = namedtuple('Feature',
                     ('embedded_feature'))

class ReplayMemory(object):
    def __init__(self, capacity, name):
        self.capacity = capacity
        self.name = name
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(self.name(*args))

    def sample(self, batch_size):
        indexes = np.random.choice(np.arange(len(self.memory)), batch_size)
        instances = [self.memory[idx] for idx in indexes]
        batch = self.name(*zip(*instances))
        return batch
    
    def compute_kernel(self, instance, neighbor_size):
        instance = instance.unsqueeze(1).to(device)
        all_instances = self.sample(len(self.memory)).embedded_feature
        instance_set = torch.cat(all_instances).unsqueeze(0).to(device)
        
        norm = torch.norm(instance - instance_set, dim=-1)
        distance = torch.pow(norm, 2)
        neighbors = -torch.topk(-distance, neighbor_size).values
        neighbors = neighbors / (neighbors.mean() + 1e-6)
        kernel = 0.0001 / (neighbors + 0.0001)
        return kernel.sum()
    
    def __len__(self):
        return len(self.memory)
    
class PER_Memory(object):
    def __init__(self, capacity, small_epsilon):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.memory_probability = deque(maxlen=capacity)
        self.small_epsilon = small_epsilon

    def push(self, *args):
        """Saves a transition.""" 
        if len(self.memory) > 0:
            max_probability = max(self.memory_probability)
        else:
            max_probability = self.small_epsilon
            
        self.memory.append(Transition(*args))
        self.memory_probability.append(max_probability)

    def sample(self, batch_size, gamma, policy_net, target_net, beta, alpha):
        probability_sum = sum(self.memory_probability)
        p = [probability / probability_sum for probability in self.memory_probability]
        
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
            self.memory_probability[idx] = pow(abs(td_error[td_error_idx]) + self.small_epsilon, alpha).item()
            td_error_idx += 1
        return batch, weights
    
    def __len__(self):
        return len(self.memory)
    
