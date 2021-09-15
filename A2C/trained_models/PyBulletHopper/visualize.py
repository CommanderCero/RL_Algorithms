import sys
sys.path.append("../../")

import utils
import models
import torch
import gym
import pybulletgym

if __name__ == "__main__":
    
    actor = utils.create_feedforward([15, 64, 128, 3])
    actor = models.GaussianPolicy(actor, (3,))
    actor.load_state_dict(torch.load("actor_latest.torch"))
    
    env_name = "HopperPyBulletEnv-v0"
    env =  gym.make(env_name)
    env.seed(1)
    
    env.render() # Call this before reset otherwise no window will show up
    utils.play(actor, env, repeats=50)