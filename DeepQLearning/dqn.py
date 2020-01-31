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
        nn.Linear(np.prod(obs_shape), 256),
        nn.ReLU(),
        nn.Linear(256, act_dim))
        
class DQNAgent:
    def __init__(self, env_fn, network_template=default_network_template,
                 reward_decay=0.99, memory_size=10000, exploration_steps=2000,
                 steps_per_epoch=250, batch_size=32):
        self.env_fn = env_fn
        self.env = env_fn()
        self.test_env = env_fn()
        self.obs_shape = self.env.observation_space.shape
        self.act_dim = self.env.action_space.n
        
        self.reward_decay = reward_decay
        self.exploration_steps = exploration_steps
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        
        self.memory = ReplayMemory(self.obs_shape, memory_size)
        self.q_network = network_template(self.obs_shape, self.act_dim)
        self.target_q_network = network_template(self.obs_shape, self.act_dim)
        self.optimizer = torch.optim.RMSprop(self.q_network.parameters(), lr=1e-3)
        
        self.step = 0
        
    def test_run(self, episodes=10, render=False):
        returns = []
        ep_lengths = []
        for i in range(episodes):
            state = self.test_env.reset()
            done = False
            returns.append(0)
            ep_lengths.append(0)
            
            while not done:
                action = self.select_action(state)
                state, reward, done, _ = self.test_env.step(action)
                if render:
                    self.test_env.render()
                
                returns[i] += reward
                ep_lengths[i] += 1
                
        # Close the test environment after rendering to prevent any "dead" windows
        if render:
            self.test_env.close()
            self.test_env = self.env_fn()
            
        return (returns, ep_lengths)
    
    def train_run(self, total_steps):
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
                state = self.env.reset()
            
            if self.step > self.exploration_steps:
                # Train the q-network
                batch = self.memory.sample(self.batch_size)
                self.train(batch)
                
                # End of epoch - Log some data about our agent
                if self.step % self.steps_per_epoch == 0:
                    returns, ep_lengths = self.test_run()
                    print("Avg Return: {:<15}Avg Ep-Length: {}".format(np.mean(returns), np.mean(ep_lengths)))
                
            
    def select_action(self, state):
        # Select action according to epsilon greedy policy
        state_tensor = torch.Tensor([state])
        with torch.no_grad():
            action = self.q_network(state_tensor)[0].argmax()
            
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
            next_q_values = self.q_network(next_states)
            target = rewards + (1 - done) * self.reward_decay * torch.max(next_q_values, axis=1).values
        
        # Calculate the q_values for the old states and the values that we want to change
        q_values = self.q_network(states)
        relevant_q_values = torch.gather(q_values, 1, actions.long().view(-1,1)).squeeze()
        
        # Train
        loss = ((relevant_q_values - target.detach()) ** 2).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        

if __name__ == "__main__":
    env_fn = lambda: gym.make("CartPole-v0")
    agent = DQNAgent(env_fn)
    
    agent.train_run(10000)
    agent.test_run(episodes=10, render=True)
