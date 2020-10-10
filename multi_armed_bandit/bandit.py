# -*- coding: utf-8 -*-
# filename: bandit.py
# brief: bandit class to simulate MAB
# author: Jia Zhuang
# date: 2020-09-23

import numpy as np

class GaussianBandit():
    def __init__(self, num_arms=5, sig=0.1, seed=None):
        self.rng = np.random.RandomState(seed=seed)
        self.num_arms = num_arms
        self.centers = self.rng.permutation(num_arms)
        self.sigma = np.ones(num_arms) * sig
    def get_reward(self, arm_id=0):
        #np.random.seed(int(time.time()))
        reward = self.rng.randn() * self.sigma[arm_id] + self.centers[arm_id]
        return reward
    
class UnstableGaussianBandit(GaussianBandit):
    def __init__(self, num_arms=5, sig=0.1, change_interval=100, change_amp=3.0, seed=None):
        super(UnstableGaussianBandit, self).__init__(num_arms=num_arms, sig=sig, seed=seed)
        self.seed = seed
        self.change_amp = change_amp
        self.rng = np.random.RandomState(seed=seed)
        self.change_interval = change_interval
        self.counter = 0
    def get_reward(self, arm_id=0):
        self.counter += 1
        if self.counter == self.change_interval:
            self.centers = self.centers + self.rng.randn(self.num_arms) * self.change_amp
            self.counter = 0
        reward = self.rng.randn() * self.sigma[arm_id] + self.centers[arm_id]
        return reward
    def reset(self):
        self.rng = np.random.RandomState(seed=self.seed)
        self.counter = 0

if __name__ == "__main__":
    toy_bandit = GaussianBandit(num_arms=8)
    print(toy_bandit.centers)
    print(toy_bandit.get_reward(1))
    print(toy_bandit.get_reward(2))