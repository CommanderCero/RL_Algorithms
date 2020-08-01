import numpy as np
import torch
import torch.nn as nn

class SimpleDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self.__get_conv_size__(input_shape)
        self.linear = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def __get_conv_size__(self, input_shape):
        output = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(output.shape))
    
    def forward(self, X):
        X = X.float() / 256 # Normalize the input
        
        output = self.conv(X).view(X.shape[0], -1)
        return self.linear(output)
    
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self.__get_conv_size__(input_shape)
        self.advantage_net = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def __get_conv_size__(self, input_shape):
        output = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(output.shape))
    
    def forward(self, X):
        X = X.float() / 256 # Normalize the input
        
        output = self.conv(X).view(X.shape[0], -1)
        value = self.value_net(output)
        advantages = self.advantage_net(output)
        
        return value + advantages - advantages.mean(1, keepdim=True)
    
        