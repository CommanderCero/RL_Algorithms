import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Policy(ABC, nn.Module):
    @abstractmethod
    def forward(self, state_batch: torch.Tensor) -> torch.distributions.Distribution:
        pass
    
    @abstractmethod
    def get_log_probs(self, dist: torch.distributions.Distribution, actions: torch.Tensor) -> torch.Tensor:
        pass
    
    @torch.no_grad()
    def get_actions(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.forward(state_batch).sample().cpu().numpy()

class SoftmaxPolicy(Policy):
    def __init__(self, logits_net: nn.Module):
        super(SoftmaxPolicy, self).__init__()
        
        self.logits_net = logits_net
        
    def forward(self, state_batch):
        logits = self.logits_net(state_batch)
        return torch.distributions.Categorical(logits=logits)
    
    def get_log_probs(self, dist: torch.distributions.Distribution, actions: torch.Tensor) -> torch.Tensor:
        return dist.log_prob(actions)
    
    
class GaussianPolicy(Policy):
    def __init__(self, mu_net: nn.Module, action_shape: Tuple[int, ...]):
        super(GaussianPolicy, self).__init__()
        
        log_std = np.full(action_shape, -0.5, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mu_net
        
    def forward(self, state_batch):
        mu = self.mu_net(state_batch)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)
    
    def get_log_probs(self, dist: torch.distributions.Distribution, actions: torch.Tensor) -> torch.Tensor:
        return dist.log_prob(actions).sum(axis=-1)
        
        
