# -*- coding: utf-8 -*-
# filename: experiments.py
# brief: experiments based on the simple RL bandit algorithms
# author: Jia Zhuang
# date: 2020-09-24

import numpy as np
import time
import matplotlib.pyplot as plt
from bandit import GaussianBandit, UnstableGaussianBandit
from MAB_algorithm_experiment import bandit_algorithm
from action_select import select_action_epsilon_greedy
from test_bandit import draw_unstable_bandit

def test_epsilon_greedy(toy_bandit, n_epoch=2000, warm_up=False, epsilon=0.1):

    q_list, aver_reward_list, act_selection_aver \
                    = bandit_algorithm(toy_bandit=toy_bandit, n_epoch=n_epoch, warm_up=warm_up, epsilon=epsilon)
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(wspace=None, hspace=0.3)
    ax1.plot(aver_reward_list)
    ax1.grid()
    ax1.set_title('Average reward vs. Iter')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Average reward')
    
    for each_arm in range(len(q_list)):
        ax2.plot(act_selection_aver[each_arm])
    ax2.grid()
    ax2.set_title('Average selection probability vs. Iter')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('arm selection probability')
    ax2.legend(['The {}th arm'.format(i) for i in range(5)], loc='upper right')
    # plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    
    ax1.bar(x=[i - 0.2 for i in range(len(q_list))], \
        height=q_list, width=0.4, color='r', label="Q value for each arm")
    ax1.bar(x=[i + 0.2 for i in range(len(q_list))], \
        height=toy_bandit.centers, width=0.4, color='b', label="True mean reward for each arm")

    ax1.grid()
    # real_mean_reward = ['arm_{}:{:.2f}'.format(i, v) for i, v in enumerate(toy_bandit.centers)]
    # ax1.set_title('real mean rewards are : \n {}'.format(' # '.join(real_mean_reward)))
    ax1.set_title('Comparison of Q value and real mean reward for each arm')
    ax1.legend()

    # plt.show()


def test_ucb_select(toy_bandit, n_epoch=200, warm_up=False, action_mode='ucb', c=1):
    q_list, aver_reward_list, act_selection_aver \
                    = bandit_algorithm(toy_bandit, n_epoch=2000, warm_up=False, action_mode='ucb', c=1)
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(wspace=None, hspace=0.3)
    ax1.plot(aver_reward_list)
    ax1.grid()
    ax1.set_title('Average reward vs. Iter')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Average reward')
    for each_arm in range(len(q_list)):
        ax2.plot(act_selection_aver[each_arm])
    ax2.grid()
    ax2.set_title('Average selection probability vs. Iter')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('arm selection probability')
    ax2.legend(['The {}th arm'.format(i) for i in range(5)], loc='upper right')
    # plt.show()


def test_unstable_bandit(unstable_bandit, n_epoch=2000, warm_up=False, epsilon=0.1, alpha=[0, 0.1]):
    aver_reward_list_all = []
    for idx, each_alpha in enumerate(alpha):
        unstable_bandit = UnstableGaussianBandit(num_arms=unstable_bandit.num_arms, \
            sig=unstable_bandit.sigma, \
            change_interval=unstable_bandit.change_interval, \
            change_amp=unstable_bandit.change_amp, \
            seed=unstable_bandit.seed)
        if idx == 0:
            draw_unstable_bandit(unstable_bandit, n_epoch=n_epoch)
            continue
        _, aver_reward_list, _ \
                        = bandit_algorithm(toy_bandit=unstable_bandit, \
                            n_epoch=n_epoch, warm_up=warm_up, epsilon=epsilon, alpha=each_alpha)
        aver_reward_list_all.append(aver_reward_list)
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111)
    fig.subplots_adjust(wspace=None, hspace=0.3)
    for each_curve in aver_reward_list_all:
        ax1.plot(each_curve)
    ax1.grid()
    ax1.legend(['alpha = {}'.format(i) for i in alpha], loc='upper right')
    ax1.set_title('Average reward vs. alpha (learning rate)')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Average reward')
    # plt.show()


if __name__ == "__main__":
    
    NUM_ARMS = 5
    SIG = 2.0
    AMP = 2.0
    INTERVAL = 200
    EPOCH = 10000
    SEED = 2020

    toy_bandit = GaussianBandit(num_arms=NUM_ARMS, sig=SIG, seed=SEED)
    print(toy_bandit.centers)
    print("Testing epsilon greedy method ... ")
    test_epsilon_greedy(toy_bandit)
    print("Testing UCB method ... ")
    test_ucb_select(toy_bandit)

    ALPHA = [0, 0.05, 0.1, 0.2, 1.0]
    # ALPHA = [0.1]
    print("Testing different alpha (learning rate) in unstable bandit ... ")
    unstable_bandit = UnstableGaussianBandit(num_arms=NUM_ARMS, sig=SIG, change_interval=INTERVAL, change_amp=AMP, seed=SEED)
    test_unstable_bandit(unstable_bandit, alpha=ALPHA, n_epoch=EPOCH)
    plt.show()
