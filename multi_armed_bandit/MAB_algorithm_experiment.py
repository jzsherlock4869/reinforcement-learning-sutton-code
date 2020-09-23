# -*- coding: utf-8 -*-
# filename: MAB_algorithm_experiment.py
# brief: testing for the simple RL bandit algorithm
# author: Jia Zhuang
# date: 2020-09-23

import numpy as np
import time
import matplotlib.pyplot as plt
from bandit import GaussianBandit
from action_select import select_action_epsilon_greedy

def bandit_algorithm(toy_bandit, n_epoch, action_mode='epsilon_greedy', warm_up=True, **params):
    num_arms = toy_bandit.num_arms
    q_list = [0] * num_arms
    num_act = [0] * num_arms
    aver_reward_list = []
    act_selection_aver = np.zeros((num_arms, n_epoch))
    cumu_reward = 0
    for i in range(n_epoch):
        if i < num_arms and warm_up:
            arm_id = i
        else:
            if action_mode == 'epsilon_greedy':
                param_dict = params
                epsilon = param_dict['epsilon']
                arm_id, _ = select_action_epsilon_greedy(q_list, epsilon=epsilon)
        num_act[arm_id] += 1
        reward = toy_bandit.get_reward(arm_id)
        q_list[arm_id] = q_list[arm_id] + 1.0 / (num_act[arm_id] + 1) * (reward - q_list[arm_id])
        
        cumu_reward += reward
        aver_reward_list.append(cumu_reward / (i + 1) * 1.0)
        for each_arm in range(num_arms):
            if each_arm == arm_id:
                act_selection_aver[each_arm, i] = act_selection_aver[each_arm, max(i-1, 0)] + 1
            else:
                act_selection_aver[each_arm, i] = act_selection_aver[each_arm, max(i-1, 0)]

    act_selection_aver = act_selection_aver / (np.array([[range(n_epoch)] * num_arms])[0,:,:] + 1)

    return q_list, aver_reward_list, act_selection_aver

if __name__ == "__main__":
    NUM_ARMS = 5
    SIG = 1.0
    toy_bandit = GaussianBandit(num_arms=NUM_ARMS, sig=SIG)
    print(toy_bandit.centers)
    q_list, aver_reward_list, act_selection_aver \
                    = bandit_algorithm(toy_bandit, n_epoch=200, warm_up=True, epsilon=0.1)
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(wspace=None, hspace=0.3)
    ax1.plot(aver_reward_list)
    ax1.grid()
    ax1.set_title('Average reward vs. Iter')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Average reward')
    for each_arm in range(NUM_ARMS):
        ax2.plot(act_selection_aver[each_arm])
    ax2.grid()
    ax2.set_title('Average selection probability vs. Iter')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('arm selection probability')
    ax2.legend(['The {}th arm'.format(i) for i in range(5)], loc='upper right')
    plt.show()
