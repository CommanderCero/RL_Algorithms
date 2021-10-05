import gym
import collections
import numpy as np
import cv2
import os
import datetime
import wandb
from pathlib import Path

import utils
import models
from parallelActorCritic import train

import torch
import torch.nn as nn

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

class OneLiveLevelOnly(gym.Wrapper):
    def __init__(self, env):
        super(OneLiveLevelOnly, self).__init__(env)
        self.prev_lives = None
        self.level = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.prev_lives is None:
            self.prev_lives = info['life']
            self.level = (info['stage'], info['world'])
        elif info['life'] < self.prev_lives:
            done = True
        elif (info['stage'], info['world']) != self.level:
            done = True
            
        return obs, reward, done, info
    
    def reset(self):
        self.prev_lives = None
        self.level = None
        return self.env.reset()

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        assert frame.shape == (240, 256, 3), "Unknown resolution."
        
        # Grayscale
        img = frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.114
        # Resize
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

def get_output_count(net: nn.Module, input_shape: tuple):
    o = net(torch.zeros(1, *input_shape))
    return int(np.prod(o.size()))

def make_env():
    env = gym.make("SuperMarioBros-v0")
    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    env = OneLiveLevelOnly(env)
    return JoypadSpace(env, SIMPLE_MOVEMENT)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--shared_layers', type=int, nargs='+', default=[32])
    parser.add_argument('--layers', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--rew_decay', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--env_count', type=int, default=1, help='The number of environments that are executed in paralel. The batch_size equals env_count * env_steps.')
    parser.add_argument('--env_steps', type=int, default=1, help='The number of steps to take for each parallel environment. The batch_size equals env_count * env_steps.')
    parser.add_argument('--steps', type=int, default=1000, help='The total number of gradient-updates until the training is considered complete.')
    parser.add_argument('--exp_name', type=str, default='actor_critic_mario')
    parser.add_argument('--log_folder', type=str, default='./logs')
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--log_video', type=bool, default=True)
    args = parser.parse_args()
    
    # Set seed for deterministic results
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Gather information about environment
    dummy_env = make_env()
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    del dummy_env
    
    # Initialize shared body
    conv_net = nn.Sequential(
        nn.Conv2d(obs_space.shape[0], 16, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=1),
        nn.ReLU()
    )
    inp_size = get_output_count(conv_net, obs_space.shape)
    shared_body = nn.Sequential(
        conv_net,
        nn.Flatten(),
        utils.create_feedforward([inp_size, *args.shared_layers])
    )
    
    # Initialize actor
    inp_size = args.shared_layers[-1]
    if isinstance(act_space, gym.spaces.Box):
        assert len(act_space.shape) == 1, "Cannot handle the environments action-space"
        net = nn.Sequential(
            shared_body,
            utils.create_feedforward([inp_size, *args.layers, act_space.shape[0]])
        )
        actor = models.GaussianPolicy(net, act_space.shape)
    elif isinstance(act_space, gym.spaces.Discrete):
        net = nn.Sequential(
            shared_body,
            utils.create_feedforward([inp_size, *args.layers, act_space.n])
        )
        actor = models.SoftmaxPolicy(net)
    else:
        raise Exception("Cannot handle the environments action-space")
        
    # Initialize critic
    critic = nn.Sequential(
        shared_body,
        utils.create_feedforward([inp_size, *args.layers, 1])
    )
    
    # Setup Logging
    full_experiment_name = f"{args.exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_folder = os.path.join(args.log_folder, full_experiment_name)
    print(f"Log Folder '{log_folder}'")
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    wandb.init(name=full_experiment_name, project='RL-Algorithms', dir=log_folder)
    
    # Train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    actor.to(device)
    critic.to(device)
    
    print("---- STARTING TRAINING ----")
    train(actor, critic, make_env, 
          train_steps=args.steps,
          device=device,
          env_count=args.env_count, env_steps=args.env_steps,
          save_frequency=args.save_freq,
          reward_decay=args.rew_decay,
          log_folder=log_folder, log_video=args.log_video)