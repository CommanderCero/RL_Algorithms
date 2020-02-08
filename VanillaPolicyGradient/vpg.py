import gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import numpy as np
import torch
import torch.nn as nn
import modules
import scipy.signal as signal
import wandb
import os

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
    def __init__(self, env_fn, reward_decay=0.99, steps_per_epoch=250, save_folder="./checkpoints"):
        super().__init__()
        
        self.env_fn = env_fn
        self.env = env_fn()
        self.test_env = env_fn()
        self.state_shape = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        
        self.reward_decay = reward_decay
        self.steps_per_epoch = steps_per_epoch
        self.save_folder = save_folder
        
        self.actor = modules.MLPGaussianPolicy(self.state_shape, self.act_dim)
        self.critic = modules.MLPValueFunction(self.state_shape)
        self.buffer = VPGBuffer(self.state_shape, self.act_dim, steps_per_epoch, self.critic)
        self.step = 0
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
    def test_run(self, episodes=10, render=False, max_steps=500):
        # Normally we would have to call render constantly while stepping through the environment
        # Pybulletgym handles this a bit differently
        # We HAVE to call it before reset and only need to do it once
        # aka -> This code only works with pybulletgym's environments
        if render:
            self.test_env.render(mode="human")
        
        returns = []
        ep_lengths = []
        for i in range(episodes):
            state = self.test_env.reset()
            returns.append(0)
            ep_lengths.append(0)
            
            for _ in range(max_steps):
                action = self.select_action(state)
                state, reward, done, _ = self.test_env.step(action)
                if render:
                    self.test_env.render(mode="human")
                
                returns[i] += reward
                ep_lengths[i] += 1
                
                if done:
                    break
                
        # Close the test environment after rendering to prevent any "dead" windows
        if render:
            self.test_env.close()
            self.test_env = self.env_fn()
            
        return (returns, ep_lengths)
    
    def train_run(self, total_steps):
        state = self.env.reset()
        episode_lengths = [0]
        episode_returns = [0]
        for i in range(total_steps):    
            # Step the environment
            action = self.select_action(state)
            new_state, reward, done, _ = self.env.step(action)
            self.step += 1
            
            # Collect some data for later logging
            episode_lengths[-1] += 1
            episode_returns[-1] += reward
            
            # Add experience to buffer
            self.buffer.add(state, action, reward)
            
            # Overwrite last observed state
            state = new_state
            
            # End of trajectory
            if done:
                self.buffer.end_trajectory(new_state, done)
                state = self.env.reset()
                
                episode_lengths.append(0)
                episode_returns.append(0)
                
            if self.step % self.steps_per_epoch == 0:
                if not done: 
                    self.buffer.end_trajectory(new_state, False)
                    
                # Train actor and critic
                data = self.buffer.get_data()
                self.buffer.clear()
                actor_loss, critic_loss = self.train(data)
                
                # Log some data
                print(np.mean(episode_returns))
                self.save_models(self.save_folder)
                
                # Weights & Biases logging
                wandb.log({
                    "Actor Loss": actor_loss,
                    "Critic Loss": critic_loss,
                    "Avg. Return": np.mean(episode_returns), 
                    "Max Return": np.max(episode_returns),
                    "Min Return": np.min(episode_returns),
                    "Avg. Episode Length": np.mean(episode_lengths)})
                
                # Clear data collection arrays
                episode_lengths = [0]
                episode_returns = [0]
                
    
    def select_action(self, state):
        state_tensor = torch.Tensor([state])
        with torch.no_grad():
            distribution = self.actor(state_tensor)
            action = distribution.sample()[0].numpy()
            
        return action
    
    def train(self, data):
        states = torch.from_numpy(data["states"])
        actions = torch.from_numpy(data["actions"])
        returns = torch.from_numpy(data["returns"])
        advantages = torch.from_numpy(data["advantages"])
        
        # Train actor
        log_probs = self.actor(states).log_prob(actions).reshape(-1)
        # Note we negate the formula to turn our minimization into an maximization
        # Also technically this is not a loss but a trick to calculate the policy gradient
        actor_loss = -(log_probs * advantages).mean()
        
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        
        # Train critic
        state_values = self.critic(states).reshape(-1)
        critic_loss = ((state_values - returns) ** 2).mean()
        
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        
        return (actor_loss, critic_loss)
    
    def save_models(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, f"actor_{self.step}.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, f"critic_{self.step}.pt"))
        
    def load_models(self, path, step):
        self.actor.load_state_dict(torch.load(os.path.join(path, f"actor_{step}.pt")))
        self.critic.load_state_dict(torch.load(os.path.join(path, f"critic_{step}.pt")))
    
if __name__ == "__main__":
    import datetime
    import argparse
    from pathlib import Path
    
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment_id", help="The ID for the OpenAI-Gym environment that will be used for training", default="InvertedPendulumPyBulletEnv-v0")
    parser.add_argument("--save_folder", help="Folder in which the model will be saved", default="./checkpoints")
    parser.add_argument("--steps_per_epoch", help="""Specifies how many steps are equal to one epoch.
                        After every epoch we will save a model checkpoint and log the agents performance""", type=int, default="500")
    parser.add_argument("--steps", help="Specifies how many steps the agent can interact with the environment", type=int, default=250000)                        
                        
    args = parser.parse_args()
    
    # Make sure the checkpoint folder exists
    Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    
    # Setup agent
    env_fn = lambda: gym.make(args.environment_id)
    agent = VPGAgent(env_fn, save_folder=args.save_folder, steps_per_epoch=args.steps_per_epoch)
    
    # Setup logging
    wandb.init(name=f"VPG_{args.environment_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}", project="rl_algorithms")
    wandb.watch(agent.actor, log="all")
    wandb.watch(agent.critic, log="all")
    
    # Train and Test
    agent.train_run(args.steps)
    
    