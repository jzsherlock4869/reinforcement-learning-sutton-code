# -*- coding: utf-8 -*-
# filename: test_bandit.py
# brief: test bandit class to simulate MAB
# author: Jia Zhuang
# date: 2020-09-24

import numpy as np
import matplotlib.pyplot as plt
from bandit import GaussianBandit, UnstableGaussianBandit

if __name__ == "__main__":

    # test normal Gaussian Bandit
    toy_bandit = GaussianBandit(num_arms=8)
    print(toy_bandit.centers)
    print(toy_bandit.get_reward(1))
    print(toy_bandit.get_reward(2))

    # test unstable Gaussian Bandit
    toy_bandit = UnstableGaussianBandit(num_arms=3, sig=2.0, change_interval=100)
    ls0, ls1, ls2 = [], [], []
    cntr0, cntr1, cntr2 = [], [], []
    for i in range(1000):
        sa = np.random.choice(3)
        r = toy_bandit.get_reward(sa)
        cntr0.append(toy_bandit.centers[0])
        cntr1.append(toy_bandit.centers[1])
        cntr2.append(toy_bandit.centers[2])
        if sa == 0:
            ls0.append(r)
        elif sa == 1:
            ls1.append(r)
        else:
            ls2.append(r)
    
    # show the change of means of each arm reward 
    # when bandit is unstable
    # reward each time the corresponding arm is selected
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(wspace=None, hspace=0.3)
    ax1.plot(ls0)
    ax1.plot(ls1)
    ax1.plot(ls2)
    ax1.grid()
    ax1.set_title('Rewards of Each Arm')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Reward')
    # reward mean change curve
    ax2.plot(cntr0)
    ax2.plot(cntr1)
    ax2.plot(cntr2)
    ax2.grid()
    ax2.set_title('Mean of Each Arm of Gaussian Bandit (showing unstability)')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Current Mean of Reward')
    plt.show()