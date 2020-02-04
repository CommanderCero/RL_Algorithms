import gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import numpy as np
import torch
import torch.nn as nn
import modules

class VPGBuffer:
    def __init__(self, obs_shape, act_dim, size):
        self.states = np.empty([size, *obs_shape], dtype=np.float32)
        self.actions = np.empty([size, act_dim], dtype=np.float32)
        self.rewards = np.empty([size], dtype=np.float32)
        self.returns = np.empty([size], dtype=np.float32)
        self.advantages = np.empty([size], dtype=np.float32)
        self.values = np.empty([size], dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        self.max_size = size
        
    def add(self):
        pass
    
class VPGAgent(nn.Module):
    def __init__(self, env_fn):
        super().__init__()
        
        self.env_fn = env_fn
        self.env = env_fn()
        self.test_env = env_fn()
        self.state_shape = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        
        self.actor = modules.MLPGaussianPolicy(self.state_shape, self.act_dim)
        self.critic = modules.MLPValueFunction(self.state_shape)
        
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
    agent.test_run(episodes=10000, render=True)
    
    