import numpy as np

class ReplayMemory:
    def __init__(self, obs_shape, act_dim, size):
        self.states = np.empty([size, *obs_shape], dtype=np.float32)
        self.actions = np.empty([size, act_dim], dtype=np.float32)
        self.next_states = np.empty([size, *obs_shape], dtype=np.float32)
        self.done_flags = np.empty([size], dtype=np.float32)
        self.rewards = np.empty([size], dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        self.max_size = size
    
    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.done_flags[self.ptr] = done
        self.rewards[self.ptr] = reward
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        idx = np.random.randint(0, self.ptr, size=batch_size)
        
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "next_states": self.next_states[idx],
            "done_flags": self.done_flags[idx],
            "rewards": self.rewards[idx]}
        
    def __len__(self):
        return self.size
        
