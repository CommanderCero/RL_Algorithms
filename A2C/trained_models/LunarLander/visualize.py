import sys
sys.path.append("../../")

import utils
import models
import torch
import gym

if __name__ == "__main__":
    
    actor = utils.create_feedforward([8, 64, 128, 2])
    actor = models.GaussianPolicy(actor, (2,))
    actor.load_state_dict(torch.load("actor_latest.torch"))
    
    env_name = "LunarLanderContinuous-v2"
    env =  gym.make(env_name)
    env.seed(1)
    utils.play(actor, env, repeats=10)