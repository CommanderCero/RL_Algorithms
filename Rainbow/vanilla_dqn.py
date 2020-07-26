import numpy as np
import ptan
import gym
import config
import torch
import torch.nn as nn

from ignite.engine import Engine

def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state  # the result will be masked anyway
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
        
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)

def compute_dqn_loss(batch, net, target_net, gamma, device):
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    
    states = torch.from_numpy(states).to(device)
    actions = torch.from_numpy(actions).to(device).unsqueeze(-1)
    rewards = torch.from_numpy(rewards).to(device)
    dones = torch.from_numpy(dones).to(device)
    next_states = torch.from_numpy(next_states).to(device)
    
    state_action_values = net(states).gather(1, actions).squeeze(-1)
    with torch.no_grad():
        next_state_action_values = target_net(next_states).max(1)[0]
        next_state_action_values[dones] = 0.0
        
    new_state_action_estimate = rewards + gamma * next_state_action_values
    return nn.MSELoss()(state_action_values, new_state_action_estimate)

def batch_generator(buffer, initial, batch_size):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)

class EpsilonTracker:
    def __init__(self, selector, eps_start, eps_end, eps_frames):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_frames = eps_frames
        
        # Set epsilon to initial value
        self.update(0)
        
    def update(self, frame):
        new_eps = self.eps_start - frame / self.eps_frames
        self.selector.epsilon = max(self.eps_end, new_eps)

if __name__ == "__main__":
    params = config.HYPERPARAMETERS["Pong"]
    
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(config.SEED)
    
    # Setup the neural networks
    net = params.model_fn(env.observation_space.shape, env.action_space.n).to(config.DEVICE)
    target_net = ptan.agent.TargetNet(net)
    print("Trainable parameters:", np.sum([np.prod(param.shape) for param in net.parameters()]))
    
    # Setup the agent
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    eps_tracker = EpsilonTracker(selector, params.epsilon_start, params.epsilon_end, params.epsilon_frames)
    agent = ptan.agent.DQNAgent(net, selector, device=config.DEVICE)
    
    # Setup experience replay
    experience_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params.gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(experience_source, buffer_size=params.replay_size)
    
    # Setup training
    optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate)
    
    def process_batch(engine, batch):
        value_loss = compute_dqn_loss(batch, net, target_net.target_model,
                                      gamma=params.gamma, device=config.DEVICE)
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()
        
        eps_tracker.update(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            target_net.sync()
            
        return {
            "loss": value_loss.item(),
            "epsilon": selector.epsilon
        }
    
    # Run Training
    engine = Engine(process_batch)
    config.setup_ignite(engine, params, experience_source, "Train")
    engine.run(batch_generator(buffer, params.replay_initial, params.batch_size))