import gym
import numpy as np

import torch
import torch.nn as nn

class ReplayMemory:
    def __init__(self, obs_shape, size):
        self.states = np.empty([size, *obs_shape], dtype=np.float32)
        self.actions = np.empty([size], dtype=np.uint8)
        self.next_states = np.empty([size, *obs_shape], dtype=np.float32)
        self.done_flags = np.empty([size], dtype=np.float32)
        self.rewards = np.empty([size], dtype=np.float32)
        
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
        nn.Linear(np.prod(obs_shape), 256),
        nn.ReLU(),
        nn.Linear(256, act_dim)
        
class DQNAgent:
    def __init__(self, env, network_template=default_network_template,
                 reward_decay=0.99, memory_size=10000, exploration_steps=2000,
                 steps_per_epoch=5000, batch_size=32):
        self.obs_shape = env.observation_space.shape
        self.act_dim = env.action_space.n
        self.env = env
        
        self.reward_decay = reward_decay
        self.exploration_steps = exploration_steps
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        
        self.memory = ReplayMemory(self.obs_shape, memory_size)
        self.q_network = network_template(self.obs_shape, self.act_dim)
        self.target_q_network = network_template(self.obs_shape, self.act_dim)
        self.step = 0
    
    def run(self, total_steps):
        state = self.env.reset()
        for i in range(total_steps):
            
            # Select action
            if self.step < self.exploration_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(state)
                
            # Step the environment
            new_state, reward, done, _ = self.env.step(action)
            self.step += 1
            
            # Add experience to memory
            self.memory.add(state, action, new_state, reward, done)
            
            # Overwrite last observed state
            state = new_state
            
            # End of trajectory
            if done:
                state = env.reset()
                
                # Train the q-network
                batch = self.memory.sample(self.batch_size)
                self.train(batch)
                
            
    def select_action(self, state):
        # Select action according to epsilon greedy policy
        state_tensor = torch.Tensor([state])
        with torch.no_grad():
            action = self.q_network.forward(state_tensor)[0].argmax()
            
        return action.item()
    
    def train(self, batch):
        states = torch.from_numpy(batch["states"])
        actions = torch.from_numpy(batch["actions"])
        next_states = torch.from_numpy(batch["next_states"])
        done = torch.from_numpy(batch["done_flags"])
        rewards = torch.from_numpy(batch["rewards"])
        
        # Calculate the target with the bellman-equation
        # Q(s,a) = r + gamma * max[Q(s', a'))]
        with torch.no_grad():
            next_q_values = self.q_network.forward(next_states)
            target = rewards + (1 - done) * self.reward_decay * torch.max(next_q_values, axis=1)
        return
        

if __name__ == "__main__":
    env = gym.make("CartPole-v0").unwrapped
    agent = DQNAgent(env)
    
    agent.run(10000)
