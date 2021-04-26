import itertools
import math
from typing import List, Tuple, Callable
import sys

import numpy as np

from nn import train


if __name__ == "__main__":
    # job_nr = int(sys.argv[1])
    job_nr = 0
    ITERATIONS = 20000
    NR_OF_JOBS = 1
    combs: List[Tuple[int, ...]]
    combs = [(1, 2, -2, -1, 1, 2, 3, 4, -2, -1, 1, 2),
             (1, 2, 3, 3, -1, 3, 1, 2, 3, 3, -1, 3),
             (1, 1, 2, 3, 1, 4, 1, 1, 2, 3, 1, 4),
             (1, 2, 3, 4, 1, 5, 1, 2, 3, 4, 1, 5)]

    bias_ls = []

    # [-8.493809700012207, 9.242773056030273, 9.242773056030273, -8.493809700012207, -8.493809700012207,
    #  9.242773056030273]
    bias_ls.append([[3.3603148460388184, -3.590137004852295], 3.797914505004883])

    # [-7.520458221435547, -5.383397579193115, -9.268661499023438, -9.268661499023438, -7.520458221435547,
    #  -9.268661499023438]
    bias_ls.append([[9.473535537719727, 2.824239730834961], -3.2367355823516846])

    # [-9.643684387207031, -9.643684387207031, -4.355031967163086, -4.360429286956787, -9.643684387207031,
    #  9.036487579345703]
    bias_ls.append([[3.0407607555389404, 6.365418434143066], -4.058375835418701])

    # [-8.817797660827637, -6.107050895690918, -4.0192437171936035, -3.9160284996032715, -8.817797660827637,
    #  7.953132629394531]
    bias_ls.append([[2.435253620147705, 5.776064395904541], -3.4490294456481934])

    inits_per_comb = [[(-8.493809700012207, 9.242773056030273)],
                      [(-7.520458221435547, -5.383397579193115, -9.268661499023438)],
                      [(-9.643684387207031, -4.355031967163086, -4.360429286956787, 9.036487579345703)],
                      [(-8.817797660827637, -6.107050895690918, -4.0192437171936035, -3.9160284996032715, -8.817797660827637)]]

    # combs = [(1, -2, 2, -1, 1, 2, 3, -4, 4, -3, 3, 4),
    #          (1, 2, 3, 3, -1, 3, 4, 5, 6, 6, -4, 6),
    #          (1, 1, 2, 3, 1, 4, 5, 5, 6, 7, 5, 8),
    #          (1, 2, 3, 4, 1, 5, 6, 7, 8, 9, 6, 10)]

    combinations = []
    initializations = []
    misclassified = []
    classes = []
    weights = []
    biases = []
    losses = []
    activations = []
    perfect_iters = []

    for i, comb in enumerate(combs):
        weight_list = list(set(map(abs, comb)))
        # inits = list(itertools.product([-0.01, 0.01], repeat=len(weight_list)))
        inits = inits_per_comb[i]
        # print(inits)

        nr_combins_per_job = int(math.ceil(len(inits) / NR_OF_JOBS))
        job_start = nr_combins_per_job * job_nr
        job_end = nr_combins_per_job * (job_nr + 1)

        print(f'OUTER {i + 1}/{len(combs)}')
        print(f'start: {job_start}, end: {job_end}')

        for j, initial in enumerate(inits):
            init_dict = dict(zip(weight_list, initial))
            print(f'Inner {i + 1}.: {j + 1}/{len(inits)}')
            miscl, cl, w, b, l, ac, pi = train(comb, init_dict, ITERATIONS)  # bias=bias_ls[i]
            print(w)
            print(b)

            combinations.append(comb)
            initializations.append(initial)
            misclassified.append(miscl)
            classes.append(cl)
            weights.append(w)
            biases.append(b)
            losses.append(l)
            activations.append(ac)
            perfect_iters.append(pi)
            print(50*'-' + '\n\n')

        break

    initializations = np.array(initializations)
    misclassified = np.array(misclassified)
    classes = np.array(classes)
    weights = np.array(weights)
    biases = np.array(biases)
    losses = np.array(losses)
    activations = np.array(activations)
    perfect_iters = np.array(perfect_iters)

    np.savez(f'data/xor_iters{ITERATIONS}_job{job_nr}_all_right.npz',
             comb=combinations,
             initial=initializations,
             miscl=misclassified,
             cl=classes,
             w=weights,
             b=biases,
             l=losses,
             act=activations,
             pi=perfect_iters)
    # np.savez('weight_sharing_xor.npz', miscl=misclassified, cl=classes, w=weights, l=losses, act=activations)

