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
    K = len(q_list)
    T = sum(t_list)
    if T == 0:
        arm_id = np.random.choice(K)
        return arm_id
    if 0 in set(t_list):
        unused_arms = [i for i, v in enumerate(t_list) if v == 0]
        arm_id = np.random.choice(unused_arms)
        return arm_id
    t_array = np.array(t_list)
    ucb_list = q_list + c * np.sqrt(np.log(T) / t_array)
    arm_id = np.argmax(ucb_list)
    return arm_id

if __name__ == "__main__":
    q_list = np.random.rand(5)
    print(q_list)

    print("test case for e-greedy strategy")
    for _ in range(5):
        arm_id, mode = select_action_epsilon_greedy(q_list, epsilon=0.2)
        print(arm_id, mode)
    
    print("test case for ucb strategy")
    t_list = [0] * len(q_list)
    for _ in range(5):
        arm_id = select_action_ucb(q_list, t_list, c=1)
        t_list[arm_id] += 1
        print(arm_id)