import gym
import numpy as np

import torch
import torch.nn as nn

class ReplayMemory:
    def __init__(self, obs_shape, act_dim, size):
        self.states = np.empty([size, *obs_shape], dtype=np.uint8)
        self.actions = np.empty([size, act_dim], dtype=np.uint8)
        self.next_states = np.empty([size, *obs_shape], dtype=np.uint8)
        self.done_flags = np.empty([size], dtype=np.uint8)
        self.rewards = np.empty([size], dtype=np.uint8)
        
        self.ptr = 0
        self.size = 0
        self.max_size = size
    
    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.done_flags[self.ptr] = done
        self.rewards[self.ptr] = reward
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "next_states": self.next_states[idx],
            "done_flags": self.done_flags[idx],
            "rewards": self.rewards[idx]}
        
    def __len__(self):
        return self.size
    
def default_network_template(obs_shape, act_dim):
    return nn.Sequential(
        nn.modules.Flatten(),
        nn.ReLU(),
        nn.Linear(np.prod(obs_shape), 256),
        nn.Linear(256, act_dim),
        nn.Softmax(dim=1)) # Calculate propabilities along the action dimension
        
class DQNAgent:
    def __init__(self, env, network_template=default_network_template,
                 memory_size=1000000, exploration_steps=20000):
        self.obs_shape = env.observation_space.shape
        self.act_dim = env.action_space.n
        
        self.env = env
        self.exploration_steps = exploration_steps
        self.memory = ReplayMemory(self.obs_shape, self.act_dim, memory_size)
        
        self.q_network = network_template(self.obs_shape, self.act_dim)
        self.target_q_network = network_template(self.obs_shape, self.act_dim)
        self.step = 0
    
    def run(self):
        state = self.env.reset()
        
        while self.step < self.exploration_steps:
            action = self.env.action_space.sample()
            new_state, reward, done, _ = self.env.step(action)
            
            self.step += 1
            self.memory.add(state, action, new_state, reward, done)
            
            if done:
                state = env.reset()
            else:
                state = new_state
            
        print("Exploration finished")
        
        
        
    def _normalize_obs_(self, obs):
        pass
        

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = DQNAgent(env)
    
    agent.run()
