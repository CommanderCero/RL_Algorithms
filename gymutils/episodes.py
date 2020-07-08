from typing import List
import gym
from collections import namedtuple

Episode = namedtuple("Episode", field_names=["observations", "actions", "rewards"])

def generate_episodes(env, agent) -> Episode:
    while True:
        episode = Episode([], [], [])
        obs = env.reset()
        done = False
        while not done:
            action = agent(obs)
            new_obs, reward, done, _ = env.step(action)
            
            episode.observations.append(obs)
            episode.actions.append(action)
            episode.rewards.append(reward)
            
            obs = new_obs
            
        yield episode
        
def generate_episode_batches(env, agent, num_episodes) -> List[Episode]:
    while True:
        episodes = []
        for episode in generate_episodes(env, agent):
            episodes.append(episode)
            
            if len(episodes) == num_episodes:
                yield episodes
                episodes = []
                
                
def visualize_episodes(env, agent, count=1):
    for _ in range(count):
        obs = env.reset()
        done = False
        while not done:
            action = agent(obs)
            obs, reward, done, _ = env.step(action)
            env.render()
        
    env.close()