# -*- coding: utf-8 -*-
# filename: MRP_value_compute.py
# brief: Compute value function for MRP process
# author: Jia Zhuang
# date: 2020-10-12

import numpy as np
from Markov_simulator import RandomMarkovRewardSimulator

def Monte_Carlo_for_MRP(markov_simu, n_tests=10, n_traject=50, gamma=0.5):
    n_states = markov_simu.num_states
    state_rewards = markov_simu.state_rewards
    V_states = [0 for _ in range(n_states)]
    print("[*] gamma is :{}, trajectory length is :{}".format(gamma, n_traject))
    # for each state, calculate its value func
    for cur_state in range(n_states):
        G = 0
        # for each monte carlo test
        print("===== evaluation for state {}".format(cur_state))
        for test_id in range(n_tests):
            g = state_rewards[cur_state]
            nxt = cur_state
            verbose_every = n_tests // 5
            for step in range(1, n_traject):
                nxt, r = markov_simu.move_on(nxt)
                g = g + r * gamma ** step
            if (test_id + 1) % verbose_every == 0:
                print(" >>> Test {} for state {}, gain is {}".format(test_id, cur_state, g))
            G = G + g
        V_states[cur_state] = G / n_tests
        print(" >>> Avg gain for state {} is {:.4f}".format(cur_state, V_states[cur_state]))
    return V_states


if __name__ == "__main__":
    rand_ms = RandomMarkovRewardSimulator(seed=2020)
    v = Monte_Carlo_for_MRP(rand_ms, n_tests=500, gamma=0.9)
    print(v)
    print(rand_ms.state_rewards)