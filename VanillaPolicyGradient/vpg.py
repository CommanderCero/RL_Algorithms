import gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import numpy as np
import torch
import torch.nn as nn
import modules
import scipy.signal as signal

def discounted_cumsum(arr, discount):
    """Taken from https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation"""
    return signal.lfilter([1], [1, -discount], x=arr[::-1])[::-1]

class VPGBuffer:
    def __init__(self, state_shape, act_dim, size, state_value_net, reward_decay=0.99):
        self.states = np.empty([size, *state_shape], dtype=np.float32)
        self.actions = np.empty([size, act_dim], dtype=np.float32)
        self.rewards = np.empty([size], dtype=np.float32)
        self.returns = np.empty([size], dtype=np.float32)
        self.advantages = np.empty([size], dtype=np.float32)
        
        self.state_value_net = state_value_net
        self.reward_decay = reward_decay
        
        self.traj_start_ptr = 0
        self.curr_ptr = 0
        self.size = 0
        self.max_size = size
        
    def end_trajectory(self, last_state, done):
        sl = slice(self.traj_start_ptr, self.curr_ptr)
        self.traj_start_ptr = self.curr_ptr
        
        # Calculate state values
        states = self.states[sl]
        states = np.append(states, last_state.reshape(1,-1), axis=0)
        with torch.no_grad():
            state_values = self.state_value_net(torch.Tensor(states))
            # Reshape for easier advantage calculation
            state_values = state_values.numpy().reshape(-1)
        
        # Collect all rewards
        rewards = self.rewards[sl]
        if not done:
            # Trajectory was cutoff before it actually finished, so we bootstrap the missing rewards
            rewards = np.append(rewards, state_values[-1])
        else:
            rewards = np.append(rewards, 0)
        
        # Calculate rewards to go
        self.returns[sl] = discounted_cumsum(rewards, self.reward_decay)[:-1]
        
        # Calculate advantage
        self.advantages[sl] = rewards[:-1] + self.reward_decay * state_values[1:] - state_values[:-1]
        
    def add(self, state, action, reward):
        assert self.size < self.max_size, "VPGBuffer overflow"
        
        self.states[self.curr_ptr] = state
        self.actions[self.curr_ptr] = action
        self.rewards[self.curr_ptr] = reward
        
        self.size += 1
        self.curr_ptr += 1
        
    def get_data(self):
        sl = slice(0, self.curr_ptr)
        return {
            "states": self.states[sl],
            "actions": self.actions[sl],
            "rewards": self.rewards[sl],
            "returns": self.returns[sl],
            "advantages": self.advantages[sl]
        }
    
    def clear(self):
        self.curr_ptr = 0
        self.traj_start_ptr = 0
        self.size = 0
        
    def __len__(self):
        return self.size
    
class VPGAgent(nn.Module):
    def __init__(self, env_fn, reward_decay=0.99, buffer_size=10000,
                 steps_per_epoch=250, batch_size=32,
                 save_folder="./checkpoints"):
        super().__init__()
        
        self.env_fn = env_fn
        self.env = env_fn()
        self.test_env = env_fn()
        self.state_shape = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        
        self.reward_decay = reward_decay
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.save_folder = save_folder
        
        self.actor = modules.MLPGaussianPolicy(self.state_shape, self.act_dim)
        self.critic = modules.MLPValueFunction(self.state_shape)
        self.buffer = VPGBuffer(self.state_shape, self.act_dim, buffer_size, self.critic)
        self.step = 0
        
    def test_run(self, episodes=10, render=False):
        # Normally we would have to call render constantly while stepping through the environment
        # Pybulletgym handles this a bit differently
        # We HAVE to call it before reset and only need to do it once
        # aka -> This code only works with pybulletgym's environments
        if render:
            self.test_env.render()
        
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
            # Step the environment
            action = self.select_action(state)
            new_state, reward, done, _ = self.env.step(action)
            self.step += 1
            
            # Add experience to buffer
            self.buffer.add(state, action, reward)
            
            # Overwrite last observed state
            state = new_state
            
            # End of trajectory
            if done:
                self.buffer.end_trajectory(new_state, done)
                state = self.env.reset()
                
            if self.step % self.steps_per_epoch == 0:
                if not done:
                    print("Warning: Cutoff trajectory")
                    self.buffer.end_trajectory(new_state, False)
                    
                data = self.buffer.get_data()
                self.buffer.clear()
            
            #if self.step > self.exploration_steps:
            #    # Train the q-network
            #    batch = self.memory.sample(self.batch_size)
            #    loss = self.train(batch)
            #    # Copy weights to target-network
            #    self.__update_target_weights__(self.copy_factor)
            #    
            #    # End of epoch - Log some data about our agent
            #    if self.step % self.steps_per_epoch == 0:
            #        returns, ep_lengths = self.test_run()
            #        print("Avg Return: {:<15.2f}".format(np.mean(returns)))
            #        
            #        # Save Model
            #        self.save_models(self.save_folder)
            #        
            #        # Weights & Biases logging
            #        wandb.log({
            #            "Loss": loss, 
            #            "Avg. Return": np.mean(returns), 
            #            "Max Return": np.max(returns),
            #            "Min Return": np.min(returns),
            #            "Avg. Episode Length": np.mean(ep_lengths),
            #            "Epsilon": epsilon})
    
    def select_action(self, state):
        state_tensor = torch.Tensor([state])
        with torch.no_grad():
            distribution = self.actor(state_tensor)
            action = distribution.sample()[0].numpy()
            
        return action
    
if __name__ == "__main__":
    # Setup agent
    env_fn = lambda: gym.make("InvertedPendulumPyBulletEnv-v0")
    agent = VPGAgent(env_fn)
    
    # Train and Test
    agent.train_run(100000)
    agent.test_run(episodes=10000, render=True)
    
    