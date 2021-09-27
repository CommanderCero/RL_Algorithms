import scipy.signal as signal
import torch
import torch.nn as nn
import numpy as np
import models
import gym
import wandb

def create_feedforward(sizes, activation=nn.ReLU): 
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)

def get_shape(shape):
    if shape is None:
        return ()
    return shape

def discounted_cumsum(rewards, reward_decay):
    """Taken from https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation"""
    return signal.lfilter([1], [1, -reward_decay], x=rewards[::-1])[::-1]

class TrajectoryBuffer:
    def __init__(self, observation_shape, action_shape, size, reward_decay=0.99):
        self.max_size = size
        self.trajectory_start = 0
        self.pos = 0
        self.reward_decay = reward_decay
        
        self.observations = np.empty((size, *observation_shape), dtype=np.float32)
        self.actions = np.empty((size, *get_shape(action_shape)), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.returns = np.empty((size,), dtype=np.float32)
        self.dones = np.empty((size,), dtype=np.float32)
        
    def store(self, observation, action, reward, done):
        assert self.pos < self.max_size, "Buffer Overflow"
        
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1
            
    def end_trajectory(self, value=0):
        # Compute return
        sl = slice(self.trajectory_start, self.pos)
        rewards = self.rewards[sl]
        rewards = np.append(rewards, value)
        self.returns[sl] = discounted_cumsum(rewards, self.reward_decay)[:-1]
        
        self.trajectory_start = self.pos
            
    def get_data(self):
        sl = slice(0, self.pos)
        data = dict(
            observations=self.observations[sl],
            actions=self.actions[sl],
            rewards=self.rewards[sl],
            returns=self.returns[sl],
            dones=self.dones[sl]
        )
        
        return {key : torch.from_numpy(value) for key, value in data.items()}
            
    def clear(self):
        self.pos = 0
        self.trajectory_start = 0
        
        
class VecTrajectoryBuffer:
    def __init__(self, observation_shape, action_shape, num_envs, size, reward_decay=0.99):
        self.max_size = size
        self.pos = 0
        self.reward_decay = reward_decay
        self.traj_starts = np.zeros((num_envs,), dtype=int)
        
        self.observations = np.empty((size, num_envs, *observation_shape), dtype=np.float32)
        self.actions = np.empty((size, num_envs, *get_shape(action_shape)), dtype=np.float32)
        self.rewards = np.empty((size, num_envs), dtype=np.float32)
        self.returns = np.empty((size, num_envs), dtype=np.float32)
        self.dones = np.empty((size, num_envs), dtype=np.float32)
        
    def store(self, observations, actions, rewards, dones):
        assert self.pos < self.max_size, "Buffer Overflow"
        
        self.observations[self.pos] = observations
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.pos += 1
        
        # Compute returns
        for env_index, done in enumerate(dones):
            if done:
                self._end_trajectory(env_index)
            
    def end_trajectory(self, values):
        for env_index, value in enumerate(values):
            self._end_trajectory(env_index, value)
        
    def _end_trajectory(self, env_index, value=0):
        # Compute return
        sl = slice(self.traj_starts[env_index], self.pos)
        rewards = self.rewards[sl, env_index]
        rewards = np.append(rewards, value)
        self.returns[sl, env_index] = discounted_cumsum(rewards, self.reward_decay)[:-1]
        
        # Update trajectory start
        self.traj_starts[env_index] = self.pos
            
    def get_data(self, device=torch.device('cpu')):
        sl = slice(0, self.pos)
        data = dict(
            observations=self._remove_env_axis(self.observations[sl]),
            actions=self._remove_env_axis(self.actions[sl]),
            rewards=self._remove_env_axis(self.rewards[sl]),
            returns=self._remove_env_axis(self.returns[sl]),
            dones=self._remove_env_axis(self.dones[sl])
        )
        
        return {key : torch.from_numpy(value).to(device) for key, value in data.items()}
            
    def clear(self):
        self.pos = 0
        self.traj_starts.fill(0)
        
    def _remove_env_axis(self, array):
        # array.shape = (size, num_envs, ???)
        shape = array.shape
        # Swap size with num_envs to ensure reshaping won't mix trajectories
        array = array.swapaxes(0, 1)
        # Flatten
        new_shape = (shape[0] * shape[1], *shape[2:])
        array = array.reshape(new_shape)
        return array
        
    
def play(model: models.Policy, env: gym.Env, repeats=10, device=torch.device('cpu')):
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            inp = torch.FloatTensor([state]).to(device)
            action = model.get_actions(inp)[0]
            state, reward, done, _ = env.step(action)
            env.render()
        
    env.close()
    
def capture_video(model: models.Policy, env: gym.Env, caption=None, fps=30):
    state = env.reset()
    done = False
    frames = []
    while not done:
        inp = torch.FloatTensor([state])
        action = model.get_actions(inp)[0]
        state, reward, done, _ = env.step(action)
        frames.append(env.render("rgb_array"))
     
    frames = np.array(frames) # (Time, Width, Height, Channels)
    frames = np.moveaxis(frames, 3, 1) # (Time, Channels, Width, Height)
    return wandb.Video(frames, caption=caption, fps=fps)