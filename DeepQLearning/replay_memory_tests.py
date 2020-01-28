from dql import ReplayMemory
import gym

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
            
            
if __name__ == "__main__":
    env = gym.make("Breakout-v0")