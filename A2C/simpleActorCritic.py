import models
import utils

import numpy as np
import gym
#import pybulletgym
import wandb
import datetime
import os
from timeit import default_timer as timer
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

def train(actor: models.Policy, critic: nn.Module, env,
          train_steps=1000, batch_size=128, reward_decay=0.99,
          actor_lr=0.0003, critic_lr=0.001,
          entropy_weight=0.01, max_gradient_norm=0.5,
          save_frequency=50, log_folder='.', log_video=False):
    
    # Initialize training
    buffer = utils.TrajectoryBuffer(env.observation_space.shape, env.action_space.shape, batch_size)
    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': actor_lr},
        {'params': critic.parameters(), 'lr': critic_lr}
    ], lr=0.001)
    start = timer()
    
    episode_lengths = [0]
    episode_returns = [0]
    log_step = 0
    state = env.reset()
    # Training loop
    for step in range(train_steps):
        ### Collect training data
        buffer.clear()
        for _ in range(batch_size):
            inp = torch.as_tensor([state], dtype=torch.float32)
            action = actor.get_actions(inp)[0]
            new_state, reward, done, _ = env.step(action)
            
            # Store data
            buffer.store(state, action, reward, done)
            episode_lengths[-1] += 1
            episode_returns[-1] += reward
            
            state = new_state
            if done:
                buffer.end_trajectory()
                episode_lengths.append(0)
                episode_returns.append(0)
                state = env.reset()
        # Bootstrap the value for the last visited state
        with torch.no_grad():
            value = critic(torch.Tensor(state)).item()
            buffer.end_trajectory(value)
        
        ### Train
        data = buffer.get_data()
        values = critic(data["observations"]).reshape(-1)
        
        # Compute Actor "Loss"
        dist = actor(data["observations"])
        log_probs = actor.get_log_probs(dist, data["actions"])
        advantages = data["returns"] - values.detach()
        actor_loss = -torch.mean(advantages * log_probs)
        
        # Compute Critic Loss
        critic_loss = F.mse_loss(values, data["returns"])
        
        # Train critic and actor -> Note we subtract entropy to encourage exploration
        entropy_loss = -torch.mean(dist.entropy())
        loss = actor_loss + critic_loss + entropy_loss * entropy_weight
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_gradient_norm)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_gradient_norm)
        optimizer.step()
        
        ### Logging
        if step % save_frequency == 0 or step == train_steps - 1:
            print(f"{step}: Avg Return={np.mean(episode_returns)}")
            
            # Save networks
            torch.save(actor.state_dict(), os.path.join(log_folder, "actor_latest.torch"))
            torch.save(critic.state_dict(), os.path.join(log_folder, "critic_latest.torch"))
            
            # Weights & Biases logging
            wandb.log({
                "Actor Loss": actor_loss,
                "Critic Loss": critic_loss,
                "Entropy Loss": entropy_loss,
                "Avg. Return": np.mean(episode_returns), 
                "Max Return": np.max(episode_returns),
                "Min Return": np.min(episode_returns),
                "Avg. Episode Length": np.mean(episode_lengths),
                "Time Passed (Minutes)": (timer() - start) / 60},
                step=log_step
            )
            
            if log_video:
                wandb.log({"Actor": utils.capture_video(actor, env)}, step=log_step)
            
            actor_params = {f"actor/param/{name}" : wandb.Histogram(param.detach()) for name, param in actor.named_parameters()}
            critic_params = {f"critic/param/{name}" : wandb.Histogram(param.detach()) for name, param in critic.named_parameters()}
            actor_grads = {f"actor/gradient/{name}" : wandb.Histogram(param.grad) for name, param in actor.named_parameters()}
            critic_grads = {f"critic/gradient/{name}" : wandb.Histogram(param.grad) for name, param in critic.named_parameters()}
            wandb.log(actor_params, step=log_step)
            wandb.log(critic_params, step=log_step)
            wandb.log(actor_grads, step=log_step)
            wandb.log(critic_grads, step=log_step)
            
            # Clear old logging data
            episode_lengths = [0]
            episode_returns = [0]
            
            # Advance log_step
            log_step += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--layers', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--rew_decay', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='actor_critic')
    parser.add_argument('--log_folder', type=str, default='./logs')
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--log_video', type=bool, default=False)
    args = parser.parse_args()
    
    # Set seed for deterministic results
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment
    env = gym.make(args.env)
    assert len(env.observation_space.shape) == 1, "Cannot handle the environments observation-space"
    obs_dim = env.observation_space.shape[0]
    
    # Initialize actor
    if isinstance(env.action_space, gym.spaces.Box):
        assert len(env.action_space.shape) == 1, "Cannot handle the environments action-space"
        net = utils.create_feedforward([obs_dim, *args.layers, env.action_space.shape[0]])
        actor = models.GaussianPolicy(net, env.action_space.shape)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        net = utils.create_feedforward([obs_dim, *args.layers, env.action_space.n])
        actor = models.SoftmaxPolicy(net)
    else:
        raise Exception("Cannot handle the environments action-space")
     
    # Initialize critic
    critic = utils.create_feedforward([obs_dim, *args.layers, 1])
    
    # Setup Logging
    full_experiment_name = f"{args.exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_folder = os.path.join(args.log_folder, full_experiment_name)
    print(f"Log Folder '{log_folder}'")
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    wandb.init(name=full_experiment_name, project='RL-Algorithms', dir=log_folder)
    
    # Train
    print("---- STARTING TRAINING ----")
    train(actor, critic, env, 
          train_steps=args.steps, 
          batch_size=args.batch_size,
          save_frequency=args.save_freq,
          reward_decay=args.rew_decay,
          log_folder=log_folder, log_video=args.log_video)