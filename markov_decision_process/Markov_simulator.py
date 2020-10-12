# -*- coding: utf-8 -*-
# filename: Markov_simulator.py
# brief: Markov reward process simulator
# author: Jia Zhuang
# date: 2020-10-12

import numpy as np
import pygraphviz as pgv
from PIL import Image


class MarkovSimulator():
    def __init__(self, trans_mat, state_values):
        assert trans_mat.shape[0] == trans_mat.shape[1]
        assert trans_mat.shape[0] == len(state_values)
        self.num_states = len(state_values)
        self.trans_mat = trans_mat
        self.state_values = state_values
    def draw_graph(self, filename):
        G = pgv.AGraph(strict=False, directed=True)
        for i in range(self.num_states):
            for j in range(self.num_states):
                G.add_edge(i, j, penwidth = self.trans_mat[i, j] * 5)
        for i in range(self.num_states):
            G.get_node(i).attr['label'] = '{}:v={}'.format(i, self.state_values[i])
        G.layout('dot')
        G.draw(filename)

class RandomMarkovSimulator(MarkovSimulator):
    def __init__(self, num_states=5, value_range=[1, 50], seed=None):
        self.rng = np.random.RandomState(seed=seed)
        rand_prob = [[self.rng.uniform(0, 2.0 / num_states) for i in range(num_states)] \
                     for j in range(num_states)]
        for j, r in enumerate(rand_prob):
            r[j] += 1 - sum(r)
        trans_mat = np.array(rand_prob)
        state_values = self.rng.randint(low=value_range[0], high=value_range[1], size=num_states)
        super(RandomMarkovSimulator, self).__init__(trans_mat=trans_mat, state_values=state_values)


if __name__ == "__main__":
    rand_ms = RandomMarkovSimulator()
    print(rand_ms.trans_mat)
    print(rand_ms.state_values)
    rand_ms.draw_graph("trans.png")
    Image.open("trans.png")
