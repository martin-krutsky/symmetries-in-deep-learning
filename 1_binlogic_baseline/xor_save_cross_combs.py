import numpy as np
import sys

from neural_net import *
from vis_err_space import *

X = np.transpose(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
Y = np.array([[0, 1, 1, 0]])

misclassificationsXOR = []
final_weightsXOR = []
entropiesXOR = []
dim_list = [2, 2, 1]

weight_combs = np.load('{}.npy'.format(int(sys.argv[1])), allow_pickle=True)

print(weight_combs.shape)

calc_cross_flat_xor(X, Y, weight_combs, int(sys.argv[1]), act_function='sigmoid')
