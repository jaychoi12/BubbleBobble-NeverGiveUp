from config import *
import torch

def get_td_error(batch_size, gamma, policy_net, target_net, state, action, next_state, reward):
    actions = tuple((map(lambda a: torch.tensor([[a]]).to(device), action))) 
    rewards = tuple((map(lambda r: torch.tensor([r]).to(device), reward))) 

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state))).to(device)
    non_final_next_states = torch.cat([s for s in next_state if s is not None]).to(device)

    state_batch = torch.cat(state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(batch_size).to(device)
    #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()   #dqn
    
    next_action = policy_net(non_final_next_states).max(1)[1].reshape(-1,1)
    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_action).squeeze().detach()   #double dqn
    expected_state_action_values = reward_batch + (gamma*next_state_values)
    
    td_error = state_action_values - expected_state_action_values.unsqueeze(1)
    
    return td_error