This repository contains many Reinforcement Learning algorithms that I've implemented over the years out of curiosity. Naturally, the algorithms are not designed to be used by other people. However, some more recent implementations are more user-friendly, such as the algorithms implemented in the **A2C** folder.

# Advantage Actor Critic
Implementation of the Actor-Critic Algorithm using Advantage Estimation to reduce the variance of the policy gradient. This implementation also includes Entropy Regularization to improve exploration.

Code can be found in **./A2C/simpleActorCritic.py**

## LunarLander
<img src="./A2C/trained_models/LunarLander/lunarLander.gif" width="400" height="auto"/>

## PyBullet Hopper
<img src="./A2C/trained_models/PyBulletHopper/hopper.gif" width="400" height="auto"/>

&NewLine;
# Parallel Advantage Actor Critic (A2C)
A parallelized version of the Advantage Actor-Critic Algorithm. Instead of exploring only one environment, we run N-Environments in parallel. As a result, we reduce the computational bottleneck created by complex environments and make much more efficient use of the GPU.

I've tested the algorithm on LunarLander and PyBullet Hopper and saw a significant reduction in computation time per step. I've also tested the algorithm on the [NES Mario environment](#https://github.com/Kautenja/gym-super-mario-bros).

Note that the agent still likes to jump into holes and enemies. The reason for this is difficult to pinpoint. It could be caused by a poorly chosen convolutional architecture, too short training time, or both. The agent also fails to learn the second level after completing the first level. The main reason for this is that the agent overfits too much on the first level, making it difficult to adapt to the second level without degrading the performance on the first level. 

<img src="./A2C/trained_models/MarioLevel1/mario.gif" width="400" height="auto"/>