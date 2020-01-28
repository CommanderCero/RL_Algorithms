from dql import ReplayMemory
import gym
import numpy as np

def collect_experiences(env, memory, num_experiences):
    state = env.reset()
    for i in range(num_experiences):
        rand_action = env.action_space.sample()
        next_state, reward, done, _ = env.step(rand_action)
        
        memory.add(state, rand_action, next_state, reward, done)
        
        if done:
            state = env.reset()
        else:
            state = next_state
            
def validate_sample_shapes(env_id):
    env = gym.make(env_id)
    memory = ReplayMemory(env.observation_space.shape, env.action_space.n, 500)
    collect_experiences(env, memory, 250)
    
    batch = memory.sample(32)
    assert batch["states"].shape == (32, *env.observation_space.shape), "Invalid sample shape"
    assert batch["actions"].shape == (32, env.action_space.n), "Invalid sample shape"
    assert batch["next_states"].shape == (32, *env.observation_space.shape), "Invalid sample shape"
    assert batch["done_flags"].shape == (32,), "Invalid sample shape"
    assert batch["rewards"].shape == (32,), "Invalid sample shape"
    
def validate_overflow_handling(env_id):
    env = gym.make(env_id)
    memory = ReplayMemory(env.observation_space.shape, env.action_space.n, 500)
    
    assert len(memory) == 0, "Invalid length"
    collect_experiences(env, memory, 250)
    assert len(memory) == 250, "Invalid length"
    collect_experiences(env, memory, 750)
    assert len(memory) == 500, "Invalid length"
    
def validate_sample_content():
    memory = ReplayMemory((100, 200), 4, 10000)
    
    state = np.random.rand(100, 200) * 50
    memory.add(state, [0,1,2,3], state, 200, 1)
    data = memory.sample(1)
    
    assert np.mean(data["states"][0] - state) <= 1e-8
    assert np.mean(data["actions"] - [0,1,2,3]) <= 1e-8
    assert np.mean(data["next_states"][0] - state) <= 1e-8
    assert int(data["rewards"][0]) == 200
    assert int(data["done_flags"][0]) == 1
            
if __name__ == "__main__":
    
    # Validate if the sampled data has the correct shape, also ensure that it works for different observation shapes
    validate_sample_shapes("Breakout-v0")
    validate_sample_shapes("CartPole-v0")
    
    # Validate that the memory can handle adding more memories than its own size
    validate_overflow_handling("Breakout-v0")
    
    # Validate the content of the sampled data
    validate_sample_content()