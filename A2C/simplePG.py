import numpy as np
import torch
import torch.nn as nn
import gym
import models
import utils

def train(policy: models.Policy, env, train_steps = 1000, reward_decay=0.99, learning_rate=0.001):
    
    action_cache = []
    state_cache = []
    reward_cache = []
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    for step in range(train_steps):
        # Clear cache
        action_cache.clear()
        state_cache.clear()
        reward_cache.clear()
        
        # Collect data from one trajectory
        state = env.reset()
        done = False
        while not done:
            inp = torch.FloatTensor([state])
            action = policy.get_actions(inp)[0]
            new_state, reward, done, _ = env.step(action)
            
            # Collect data
            action_cache.append(action)
            state_cache.append(state)
            reward_cache.append(reward)
            
            state = new_state
            
        # Compute "Loss"-function for computing the policy gradient
        rewards_to_go = np.array(utils.discounted_cumsum(reward_cache, reward_decay))
        dist = policy(torch.Tensor(state_cache))
        log_probs = policy.get_log_probs(dist, torch.Tensor(action_cache))
        loss = torch.mean(-log_probs * torch.Tensor(rewards_to_go.reshape(log_probs.shape)))
        
        # Gradient descent (Technically ascend since we took the negative of the policy gradient)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Loss={loss}\tReward Sum={np.sum(rewards_to_go)}")

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    
    policy = models.SoftmaxPolicy(nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ))
    
    train(policy, env)
    
    env = gym.make("CartPole-v0")
    utils.play(policy, env)