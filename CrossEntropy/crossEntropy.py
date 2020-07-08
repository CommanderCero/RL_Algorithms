import sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.nn import Sequential, Linear, ReLU
from torch.nn.functional import softmax
from gymutils.episodes import generate_episode_batches, visualize_episodes

def create_one_layer_net(num_inputs, num_outputs, hidden_size):
    return Sequential(
        Linear(num_inputs, hidden_size),
        ReLU(),
        Linear(hidden_size, num_outputs)
    )

class SoftmaxAgent:
    def __init__(self, brain):
        self.brain = brain
    
    def __call__(self, obs):
        obs = torch.FloatTensor(obs)
        probs = softmax(self.brain(obs), dim=0).detach().numpy()
        return np.random.choice(len(probs), p=probs)

def get_elite_episodes(episodes, upper_quantile=0.7):
    rewards = [np.sum(ep.rewards) for ep in episodes]
    limit = np.quantile(rewards, upper_quantile)
    
    return [episodes[i] for i in range(len(episodes)) if rewards[i] >= limit]

if __name__ == "__main__":
    import gym
    
    # Init everything
    env = gym.make("CartPole-v1")
    net = create_one_layer_net(env.observation_space.shape[0], env.action_space.n, 128)
    optimizer = Adam(net.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    agent = SoftmaxAgent(net)
    
    # Train
    counter = 0
    for i, batch in enumerate(generate_episode_batches(env, agent, 16)):
        filtered_batch = get_elite_episodes(batch)
        
        # Convert the batch into a form suitable for the network
        observations = torch.FloatTensor([obs for ep in filtered_batch for obs in ep.observations])
        actions = torch.LongTensor([act for ep in filtered_batch for act in ep.actions])
        
        # Compute loss and update weights
        logits = net(observations)
        loss = loss_func(logits, actions)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Check if we are done
        mean_reward = np.mean([np.sum(ep.rewards) for ep in batch])
        print(f"{i}: Loss: {loss.detach().item()}\tMean Reward: {mean_reward}")
        if mean_reward > 199:
            break
        
    visualize_episodes(env, agent, count=5)