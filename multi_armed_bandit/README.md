# Multi-armed Bandit 

**Multi-armed bandit (MAB)** is a quite simple and easy to implement toy problem for understanding the concept of reinforcement learning. The codes here solve the MAB by epsilon-greedy strategy and UCB (Upper Confidence Bound) strategy for action selection, using the straightforward reward expectation as the Q value.

## Preliminary results
### 5-armed stable gaussian bandit training for 200 epoches (using epsilon greedy action selection).

![egreedy_result](./result_pics/egreedy_result.png)

### Q values estimated and real reward mean.

![egreedy_q_values](./result_pics/q_values.png)

### different epsilons for epsilon-greedy

![diff_epsilon](./result_pics/diff_epsilon.png)

### using UCB for action selection.

![ucb_result](./result_pics/ucb_result.png)

### solving unstable gaussian bandit using given stride alpha to enhance the weight of nearby rewards.

the unstable gaussian bandit is shown below, indicating the mean(reward) of each arm all changes with time.
![unstable_mab](./result_pics/unstable_mab.png)

comparison alpha stride and common (1/n) stride for unstable MAB problem



