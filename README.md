# Reinforcement_Learning_Toys
Some reinforcement learning algorithm implementations. Toy models ~

# Contents 

|  Algorithms  |  Links  |
|  :----:  | :----: |
|  Multi-armed bandit  | ![link](https://github.com/jzsherlock4869/Reinforcement_Learning_Toys/tree/master/multi_armed_bandit) |
| Markov decision process  | link |

# Detailed introduction

## Multi-armed Bandits

[multi-armed bandit](./multi_armed_bandit/result_pics/mab_comic.jpg)

Multi-armed bandit problem (MAB) is a simple and fundamental example for reinforcement learning, and has been used in real world tasks (recommender sys etc.).

Defination ( from wiki ):

In probability theory, the multi-armed bandit problem (sometimes called the K- or N-armed bandit problem) is a problem in which a fixed limited set of resources must be allocated between competing (alternative) choices in a way that maximizes their expected gain, when each choice's properties are only partially known at the time of allocation, and may become better understood as time passes or by allocating resources to the choice. [wiki:multi-armed bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit)

In the MAB problem, agent uses the previous reward in the earlier actions to estimate the value of each arm, and try to maximize the expected gain for each action.



