# -*- coding: utf-8 -*-
# filename: grid_world_simu.py
# brief: Simulator of GridWorld of R.S.Sutton's textbook example 3.5
# author: Jia Zhuang
# date: 2020-10-13

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GridWorldSimulator():
    def __init__(self, H=5, W=5, 
                 bonus_coor=[(0,1), (0,3)], 
                 dest_coor=[(4,1), (2,3)], 
                 bonus=[10, 5]):
        assert len(bonus_coor) == len(set(bonus_coor))
        for point_id in range(len(bonus_coor)):
            assert bonus_coor[point_id][0] < H and bonus_coor[point_id][1] < W
            assert dest_coor[point_id][0] < H and dest_coor[point_id][1] < W
        self.H = H
        self.W = W
        self.bonus_move = dict()
        for point_id in range(len(bonus_coor)):
            self.bonus_move[bonus_coor[point_id]] = dest_coor[point_id]
        # 4 actions for left/right/up/down
        action_reward = np.zeros((H, W, 4))
        # edge reward is -1.0
        action_reward[0, :, 2] = -1.0
        action_reward[:, 0, 0] = -1.0
        action_reward[:, W - 1, 1] = -1.0
        action_reward[H - 1, :, 3] = -1.0
        for point_id in range(len(bonus_coor)):
            x, y = bonus_coor[point_id]
            action_reward[x, y, :] = bonus[point_id]
        self.sa_reward_map = action_reward
    
    def take_move(self, s=(0,1), action=0):
        action_code = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        if (s[0] == 0 and action == action_code['up']) \
            or (s[1] == 0 and action == action_code['left']) \
            or (s[0] == self.H - 1 and action == action_code['down']) \
            or (s[1] == self.W - 1 and action == action_code['right']):
            nxt_s = s
        if s in self.bonus_move:
            nxt_s = self.bonus_move[s]
        else:
            if action == action_code['left']:
                nxt_s = (s[0], s[1] - 1)
            elif action == action_code['right']:
                nxt_s = (s[0], s[1] + 1)
            elif action == action_code['up']:
                nxt_s = (s[0] - 1, s[1])
            else: #action == action_code['down']
                nxt_s = (s[0] + 1, s[1])

        reward = self.sa_reward_map[s[0], s[1], action]
        return nxt_s, reward


if __name__ == "__main__":

    my_grid = GridWorldSimulator()
    print(my_grid.take_move(s=(0,1), action=1))
    print(my_grid.take_move(s=(2,1), action=3))
    print(my_grid.take_move(s=(4,1), action=3))

    left_reward = my_grid.sa_reward_map[:, :, 0]
    right_reward = my_grid.sa_reward_map[:, :, 1]
    up_reward = my_grid.sa_reward_map[:, :, 2]
    down_reward = my_grid.sa_reward_map[:, :, 3]
    fig = plt.figure(figsize=(20, 4))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    fig.subplots_adjust(wspace=None, hspace=0.5)
    sns.heatmap(left_reward, ax=ax1,annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size':12})
    ax1.set_title('move left')
    sns.heatmap(right_reward, ax=ax2, annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size':12})
    ax2.set_title('move right')
    sns.heatmap(up_reward, ax=ax3, annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size':12})
    ax3.set_title('move up')
    sns.heatmap(down_reward, ax=ax4, annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size':12})
    ax4.set_title('move down')
    plt.show()