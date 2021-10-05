import sys
sys.path.append("../../")

import utils
import models
import torch
import torch.nn as nn
import numpy as np
import gym
from superMarioA2C import make_env

def get_output_count(net: nn.Module, input_shape: tuple):
    o = net(torch.zeros(1, *input_shape))
    return int(np.prod(o.size()))

if __name__ == "__main__":
    env = make_env()
    
    # Load actor
    conv_net = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=1),
        nn.ReLU()
    )
    inp_size = get_output_count(conv_net, env.observation_space.shape)
    shared_body = nn.Sequential(
        conv_net,
        nn.Flatten(),
        utils.create_feedforward([inp_size, 32])
    )
    inp_size = 32
    net = nn.Sequential(
        shared_body,
        utils.create_feedforward([inp_size, 64, 128, env.action_space.n])
    )
    actor = models.SoftmaxPolicy(net)
    actor.load_state_dict(torch.load("actor_latest.torch", map_location=torch.device('cpu')))
    
    # Run
    env = make_env()
    utils.play(actor, env, repeats=10)