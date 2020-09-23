# -*- coding: utf-8 -*-
# filename: bandit.py
# brief: bandit class to simulate MAB
# author: Jia Zhuang
# date: 2020-09-23

import numpy as np

class GaussianBandit():
    def __init__(self, num_arms=5, sig=0.1):
        self.num_arms = num_arms
        self.centers = np.random.permutation(num_arms)
        self.sigma = np.ones(num_arms) * sig
    def get_reward(self, arm_id=0):
        reward = np.random.randn() * self.sigma[arm_id] + self.centers[arm_id]
        return reward

if __name__ == "__main__":
    toy_bandit = GaussianBandit(num_arms=8)
    print(toy_bandit.centers)
    print(toy_bandit.get_reward(1))
    print(toy_bandit.get_reward(2))