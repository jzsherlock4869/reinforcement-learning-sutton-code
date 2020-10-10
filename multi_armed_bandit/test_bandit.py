# -*- coding: utf-8 -*-
# filename: test_bandit.py
# brief: test bandit class to simulate MAB
# author: Jia Zhuang
# date: 2020-09-24

import numpy as np
import matplotlib.pyplot as plt
from bandit import GaussianBandit, UnstableGaussianBandit

def draw_unstable_bandit(toy_bandit, n_epoch=0):
    num_arms = toy_bandit.num_arms
    change_interval = toy_bandit.change_interval
    ls = [[] for _ in range(num_arms)]
    cntr = [[] for _ in range(num_arms)]
    if n_epoch == 0:
        n_epoch = 10 * change_interval
    for _ in range(n_epoch):
        for arm in range(num_arms):
            cntr[arm].append(toy_bandit.centers[arm])
        sa = np.random.choice(num_arms)
        r = toy_bandit.get_reward(sa)
        ls[sa].append(r)
    # show the change of means of each arm reward 
    # when bandit is unstable
    # reward each time the corresponding arm is selected

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(wspace=None, hspace=0.3)
    for i in range(num_arms):
        ax1.plot(ls[i])
    ax1.grid()
    ax1.set_title('Rewards of Each Arm')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Reward')
    # reward mean change curve
    for i in range(num_arms):
        ax2.plot(cntr[i])
    ax2.grid()
    ax2.set_title('Mean of Each Arm of Gaussian Bandit (showing unstability)')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Current Mean of Reward')
    # plt.show()


if __name__ == "__main__":

    # test normal Gaussian Bandit
    toy_bandit = GaussianBandit(num_arms=8)
    print(toy_bandit.centers)
    print(toy_bandit.get_reward(1))
    print(toy_bandit.get_reward(2))

    # test unstable Gaussian Bandit
    toy_bandit = UnstableGaussianBandit(num_arms=3, sig=2.0, change_interval=100)
    draw_unstable_bandit(toy_bandit)
    plt.show()
