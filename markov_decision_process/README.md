# Markov Reward Process (MRP) and Markov Decision Process (MDP)

Both MRP and MDP obey Markovian property, i.e. "the future is independent of the past if present state is given".

## Markov Reward Process (MRP) : the state transition is independent of our actions
Markov reward model or Markov reward process is a stochastic process which extends either a Markov chain or continuous-time Markov chain by adding a reward rate to each state. (from [wiki](https://en.wikipedia.org/wiki/Markov_reward_model)).

In MRP, each state returns a reward, and the transition of states are only related to the current state.

The transition probability is:

<img src="http://latex.codecogs.com/gif.latex?P(s_{t+1} = S_j | s_t = S_i) ">

[//]: # ($$ P(s_{t+1} = S_j | s_t = S_i) $$)

The reward of each state is defined as a one-param function:

<img src="http://latex.codecogs.com/gif.latex?R(s_t = S_i) = E[r_t | s_t = S_i]">

[//]: # ($$ R(s_t = S_i) = E[r_t | s_t = S_i] $$)


## Markov Decision Process (MDP) : the state transition controlled by current state and action.

Markov decision process (MDP) is a discrete-time stochastic control process. It provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker. (from [wiki](https://en.wikipedia.org/wiki/Markov_decision_process)) 

The transition probability from state S_i to S_j under action A_k is defined as follows:

<img src="http://latex.codecogs.com/gif.latex?P(s_{t+1} = S_j | s_t = S_i, a_t = A_k) ">

[//]: # ($$ P(s_{t+1} = S_j | s_t = S_i, a_t = A_k) $$)

The reward function of MDP has two parameters:

<img src="http://latex.codecogs.com/gif.latex?R(s_t = S_i, a = A_k) = E[r_{t+1} | s_t = S_i, a = A_k]">

[//]: # ($$ R(s_t = S_i, a = A_k) = E[r_{t+1} | s_t = S_i, a = A_k] $$)


