import os
import gym
import pybulletgym  # register PyBullet enviroments with open ai gym
from vpg import VPGAgent
import matplotlib.pyplot as plt
from matplotlib import animation

def collect_frames(agent, num_frames):
    env = agent.env
    state = env.reset()
    
    frames = []
    for i in range(num_frames):
        state, _, _, _ = env.step(agent.select_action(state))
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        
    return frames

if __name__ == "__main__":
    NUM_FRAMES = 200
    target_envs = ["InvertedPendulumPyBulletEnv-v0",
                   "InvertedDoublePendulumPyBulletEnv-v0",
                   "Walker2DPyBulletEnv-v0",
                   "HopperPyBulletEnv-v0",
                   "AntPyBulletEnv-v0",
                   "HumanoidFlagrunPyBulletEnv-v0"]
    
    # Collect frames
    all_frames = []
    for env in target_envs:
        print(f"Start collecting frames for {env}")
        env_fn = lambda: gym.make(env).unwrapped
        agent = VPGAgent(env_fn)
        agent.load_models(f"./trained_models/{env}", 2000000)
        
        frames = collect_frames(agent, NUM_FRAMES)
        all_frames.append(frames)
        print(f"Finished collecting frames for {env}")
        
    # Create a gif showing all environments at the same time
    fig = plt.figure()
    axes = fig.subplots(2, 3).reshape(-1)
    patches = []
    for ax in axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        patches.append(ax.imshow(all_frames[0][0]))
    
    def animate(frame_index):
        for i, frames in enumerate(all_frames):
            patches[i].set_data(frames[frame_index])
    
    print("Start generating animation")
    anim = animation.FuncAnimation(fig, animate, frames=NUM_FRAMES, interval=20)
    print("Saving animation as gif")
    anim.save("./test.gif", writer='imagemagick', fps=60)
    