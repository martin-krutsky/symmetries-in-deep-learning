import numpy as np
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn.functional as F


goal_state = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]
all_goal_states_ls = [
    goal_state[0] + goal_state[1] + goal_state[2] + goal_state[3] + goal_state[4] + goal_state[5],
    goal_state[2] + goal_state[3] + goal_state[1] + goal_state[0] + goal_state[4] + goal_state[5],
    goal_state[1] + goal_state[0] + goal_state[3] + goal_state[2] + goal_state[4] + goal_state[5],
    goal_state[3] + goal_state[2] + goal_state[0] + goal_state[1] + goal_state[4] + goal_state[5],

    goal_state[0] + goal_state[1] + goal_state[3] + goal_state[2] + goal_state[5] + goal_state[4],
    goal_state[3] + goal_state[2] + goal_state[1] + goal_state[0] + goal_state[5] + goal_state[4],
    goal_state[1] + goal_state[0] + goal_state[2] + goal_state[3] + goal_state[5] + goal_state[4],
    goal_state[2] + goal_state[3] + goal_state[0] + goal_state[1] + goal_state[5] + goal_state[4],

    goal_state[0] + goal_state[1] + goal_state[5] + goal_state[4] + goal_state[2] + goal_state[3],
    goal_state[5] + goal_state[4] + goal_state[1] + goal_state[0] + goal_state[2] + goal_state[3],
    goal_state[1] + goal_state[0] + goal_state[4] + goal_state[5] + goal_state[2] + goal_state[3],
    goal_state[4] + goal_state[5] + goal_state[0] + goal_state[1] + goal_state[2] + goal_state[3],

    goal_state[0] + goal_state[1] + goal_state[4] + goal_state[5] + goal_state[3] + goal_state[2],
    goal_state[4] + goal_state[5] + goal_state[1] + goal_state[0] + goal_state[3] + goal_state[2],
    goal_state[1] + goal_state[0] + goal_state[5] + goal_state[4] + goal_state[3] + goal_state[2],
    goal_state[5] + goal_state[4] + goal_state[0] + goal_state[1] + goal_state[3] + goal_state[2],

    goal_state[5] + goal_state[4] + goal_state[2] + goal_state[3] + goal_state[0] + goal_state[1],
    goal_state[2] + goal_state[3] + goal_state[4] + goal_state[5] + goal_state[0] + goal_state[1],
    goal_state[4] + goal_state[5] + goal_state[3] + goal_state[2] + goal_state[0] + goal_state[1],
    goal_state[3] + goal_state[2] + goal_state[5] + goal_state[4] + goal_state[0] + goal_state[1],

    goal_state[5] + goal_state[4] + goal_state[3] + goal_state[2] + goal_state[1] + goal_state[0],
    goal_state[3] + goal_state[2] + goal_state[4] + goal_state[5] + goal_state[1] + goal_state[0],
    goal_state[4] + goal_state[5] + goal_state[2] + goal_state[3] + goal_state[1] + goal_state[0],
    goal_state[2] + goal_state[3] + goal_state[5] + goal_state[4] + goal_state[1] + goal_state[0],
]
all_goal_states_np = np.array(all_goal_states_ls)

nongoal_state = [[0, 1, 2, 3], [0, 1, 2, 3], [2, 3, 4, 5], [2, 3, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5]]
all_nongoal_states_ls = [
    nongoal_state[0] + nongoal_state[1] + nongoal_state[2] + nongoal_state[3] + nongoal_state[4] + nongoal_state[5],
    nongoal_state[2] + nongoal_state[3] + nongoal_state[1] + nongoal_state[0] + nongoal_state[4] + nongoal_state[5],
    nongoal_state[1] + nongoal_state[0] + nongoal_state[3] + nongoal_state[2] + nongoal_state[4] + nongoal_state[5],
    nongoal_state[3] + nongoal_state[2] + nongoal_state[0] + nongoal_state[1] + nongoal_state[4] + nongoal_state[5],

    nongoal_state[0] + nongoal_state[1] + nongoal_state[3] + nongoal_state[2] + nongoal_state[5] + nongoal_state[4],
    nongoal_state[3] + nongoal_state[2] + nongoal_state[1] + nongoal_state[0] + nongoal_state[5] + nongoal_state[4],
    nongoal_state[1] + nongoal_state[0] + nongoal_state[2] + nongoal_state[3] + nongoal_state[5] + nongoal_state[4],
    nongoal_state[2] + nongoal_state[3] + nongoal_state[0] + nongoal_state[1] + nongoal_state[5] + nongoal_state[4],

    nongoal_state[0] + nongoal_state[1] + nongoal_state[5] + nongoal_state[4] + nongoal_state[2] + nongoal_state[3],
    nongoal_state[5] + nongoal_state[4] + nongoal_state[1] + nongoal_state[0] + nongoal_state[2] + nongoal_state[3],
    nongoal_state[1] + nongoal_state[0] + nongoal_state[4] + nongoal_state[5] + nongoal_state[2] + nongoal_state[3],
    nongoal_state[4] + nongoal_state[5] + nongoal_state[0] + nongoal_state[1] + nongoal_state[2] + nongoal_state[3],

    nongoal_state[0] + nongoal_state[1] + nongoal_state[4] + nongoal_state[5] + nongoal_state[3] + nongoal_state[2],
    nongoal_state[4] + nongoal_state[5] + nongoal_state[1] + nongoal_state[0] + nongoal_state[3] + nongoal_state[2],
    nongoal_state[1] + nongoal_state[0] + nongoal_state[5] + nongoal_state[4] + nongoal_state[3] + nongoal_state[2],
    nongoal_state[5] + nongoal_state[4] + nongoal_state[0] + nongoal_state[1] + nongoal_state[3] + nongoal_state[2],

    nongoal_state[5] + nongoal_state[4] + nongoal_state[2] + nongoal_state[3] + nongoal_state[0] + nongoal_state[1],
    nongoal_state[2] + nongoal_state[3] + nongoal_state[4] + nongoal_state[5] + nongoal_state[0] + nongoal_state[1],
    nongoal_state[4] + nongoal_state[5] + nongoal_state[3] + nongoal_state[2] + nongoal_state[0] + nongoal_state[1],
    nongoal_state[3] + nongoal_state[2] + nongoal_state[5] + nongoal_state[4] + nongoal_state[0] + nongoal_state[1],

    nongoal_state[5] + nongoal_state[4] + nongoal_state[3] + nongoal_state[2] + nongoal_state[1] + nongoal_state[0],
    nongoal_state[3] + nongoal_state[2] + nongoal_state[4] + nongoal_state[5] + nongoal_state[1] + nongoal_state[0],
    nongoal_state[4] + nongoal_state[5] + nongoal_state[2] + nongoal_state[3] + nongoal_state[1] + nongoal_state[0],
    nongoal_state[2] + nongoal_state[3] + nongoal_state[5] + nongoal_state[4] + nongoal_state[1] + nongoal_state[0],
]
all_nongoal_states_np = np.array(all_nongoal_states_ls)


incidence_matrix = np.array([[0, 1, 0, 2, 0, 3, 0, 13, 0, 18, 1, 2, 1, 3, 1, 4, 1, 19, 2, 3, 2, 15, 2, 20, 3, 6, 3, 21,
                              4, 5, 4, 6, 4, 7, 4, 19, 5, 6, 5, 7, 5, 8, 5, 17, 6, 7, 6, 21, 7, 10, 7, 23, 8, 9, 8, 10,
                              8, 11, 8, 17, 9, 10, 9, 11, 9, 12, 9, 16, 10, 11, 10, 23, 11, 14, 11, 22, 12, 13, 12, 14,
                              12, 15, 12, 16, 13, 14, 13, 15, 13, 18, 14, 15, 14, 22, 15, 20, 16, 17, 16, 18, 16, 19,
                              17, 18, 17, 19, 18, 19, 20, 21, 20, 22, 20, 23, 21, 22, 21, 23, 22, 23],
                             [1, 0, 2, 0, 3, 0, 13, 0, 18, 0, 2, 1, 3, 1, 4, 1, 19, 1, 3, 2, 15, 2, 20, 2, 6, 3, 21, 3,
                              5, 4, 6, 4, 7, 4, 19, 4, 6, 5, 7, 5, 8, 5, 17, 5, 7, 6, 21, 6, 10, 7, 23, 7, 9, 8, 10, 8,
                              11, 8, 17, 8, 10, 9, 11, 9, 12, 9, 16, 9, 11, 10, 23, 10, 14, 11, 22, 11, 13, 12, 14, 12,
                              15, 12, 16, 12, 14, 13, 15, 13, 18, 13, 15, 14, 22, 14, 20, 15, 17, 16, 18, 16, 19, 16,
                              18, 17, 19, 17, 19, 18, 21, 20, 22, 20, 23, 20, 22, 21, 23, 21, 23, 22]])

edge_tps = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 3, 3,
            2, 2, 1, 1, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 3, 3, 3, 3, 1, 1, 3, 3,
            3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
            1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1]
edge_tps_np = np.array(edge_tps)
# edge_tps_np = (-1 * edge_tps_np + 4)
edge_tps_np = 1/edge_tps_np


