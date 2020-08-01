import config
import ptan
import gym
import torch
import numpy as np

if __name__ == "__main__":
    params = config.HYPERPARAMETERS["Pong"]
    model_weights_path = "checkpoints/n_step_dqn.torch"
    
    # Setup environment
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(config.SEED)
    
    # Create network and load weights
    net = params.model_fn(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(model_weights_path))
    
    # Create agent with greedy action selection
    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, selector)
    
    # Run environment and render it
    state = env.reset()
    done = False
    while not done:
        action_batch, _ = agent(np.expand_dims(np.array(state), 0))
        state, _, _, _ = env.step(action_batch[0])
        env.render()
    env.close()