import torch
import torch.nn as nn
import math
import numpy as np

class MLPGaussianPolicy(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super().__init__()
        
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        
        self.log_std = np.ones(act_dim, dtype=np.float32) * -0.5
        self.log_std = nn.Parameter(torch.from_numpy(self.log_std))
        self.mean_network = nn.Sequential(
            nn.modules.Flatten(),
            nn.Linear(np.prod(obs_shape), 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim))
        
    def forward(self, obs, actions=None):
        mean = self.mean_network(obs)
        std = math.e ** self.log_std
        
        return torch.distributions.Normal(mean, std)
    
class MLPValueFunction(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        
        self.value_network = nn.Sequential(
            nn.modules.Flatten(),
            nn.Linear(np.prod(obs_shape), 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
    
    def forward(self, obs):
        return self.value_network(obs)
        
        