# -*- coding: utf-8 -*-
# filename: Markov_simulator.py
# brief: Markov reward process simulator
# author: Jia Zhuang
# date: 2020-10-12

import numpy as np
import pygraphviz as pgv
from PIL import Image


class MarkovRewardSimulator():
    def __init__(self, trans_mat, state_rewards):
        assert trans_mat.shape[0] == trans_mat.shape[1]
        assert trans_mat.shape[0] == len(state_rewards)
        self.num_states = len(state_rewards)
        self.trans_mat = trans_mat
        self.state_rewards = state_rewards
    def draw_graph(self, filename):
        G = pgv.AGraph(strict=False, directed=True)
        for i in range(self.num_states):
            for j in range(self.num_states):
                G.add_edge(i, j, penwidth=self.trans_mat[i,j] * 5)
        for i in range(self.num_states):
            G.get_node(i).attr['label'] = '{}:v={}'.format(i, self.state_rewards[i])
        G.layout('dot')
        G.draw(filename)
    def move_on(self, cur_state):
        trans_prob = self.trans_mat[cur_state, :]
        next_state = np.random.choice(list(range(self.num_states)), p=trans_prob)
        reward = self.state_rewards[next_state]
        return next_state, reward

class RandomMarkovRewardSimulator(MarkovRewardSimulator):
    def __init__(self, num_states=5, value_range=[1, 50], seed=None):
        self.rng = np.random.RandomState(seed=seed)
        rand_prob = self.rng.uniform(0, 1, size=(num_states, num_states))
        # rand_prob = self.rng.randn(num_states, num_states)
        rand_prob = np.abs(rand_prob)
        for i in range(num_states):
            rand_prob[i, :] = rand_prob[i, :] / np.sum(rand_prob[i, :])
        state_rewards = self.rng.randint(low=value_range[0], high=value_range[1], size=num_states)
        super(RandomMarkovRewardSimulator, self).__init__(trans_mat=rand_prob, state_rewards=state_rewards)


if __name__ == "__main__":

    SEED = 2020

    rand_ms = RandomMarkovRewardSimulator(seed=SEED)
    print(rand_ms.trans_mat)
    print(rand_ms.state_rewards)
    rand_ms.draw_graph("trans.png")
    img = Image.open("trans.png")
    img.show()

    nxt = 0
    for _ in range(5):
        nxt, r = rand_ms.move_on(nxt)
        print("current state:{}, reward:{}".format(nxt, r))
    