# -*- coding: utf-8 -*-
# filename: action_select.py
# brief: action selection strategies in MAB problem
# author: Jia Zhuang
# date: 2020-09-23

import numpy as np
import time

def select_action_epsilon_greedy(q_list, epsilon=0.3):
    prob = np.random.rand()
    if prob > epsilon:
        arm_id = np.argmax(q_list)
        mode = 0
    else:
        arm_id = np.random.choice(len(q_list))
        mode = 1
    # mode 0 for exploitation, mode 1 for exploration
    return arm_id, mode

def select_action_ucb(q_list, t_list, c=1):
    # todo
    arm_id = 0
    return arm_id

if __name__ == "__main__":
    q_list = np.random.rand(5)
    print(q_list)
    for _ in range(5):
        arm_id, mode = select_action_epsilon_greedy(q_list, epsilon=0.2)
        print(arm_id, mode)