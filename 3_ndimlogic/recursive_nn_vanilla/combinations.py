import itertools
import math
from typing import List, Tuple, Callable
import sys

import numpy as np

from nn import train


if __name__ == "__main__":
    # job_nr = int(sys.argv[1])
    job_nr = 0

    ITERATIONS = 2000
    # NR_OF_JOBS = 100
    NR_OF_JOBS = 1
    comb: Tuple[int, ...]
    comb = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    weight_list = list(set(map(abs, comb)))
    inits = list(itertools.product([-0.01, 0.01], repeat=len(weight_list)))

    nr_combins_per_job = int(math.ceil(len(inits) / NR_OF_JOBS))
    job_start = nr_combins_per_job * job_nr
    job_end = nr_combins_per_job * (job_nr + 1)
    # job_start = rozdeleni[job_nr][0]
    # job_end = rozdeleni[job_nr][1]

    print(f'job start: {job_start}')
    print(f'job end: {job_end}')

    combinations = []
    initializations = []
    misclassified = []
    classes = []
    weights = []
    biases = []
    losses = []
    activations = []
    perfect_iters = []

    for j, initial in enumerate(inits[job_start:job_end]):
        init_dict = dict(zip(weight_list, initial))
        print(f'{j + 1}/{len(inits)}')
        miscl, cl, w, b, l, ac, pi = train(comb, init_dict, ITERATIONS)
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

    initializations = np.array(initializations)
    misclassified = np.array(misclassified)
    classes = np.array(classes)
    weights = np.array(weights)
    biases = np.array(biases)
    losses = np.array(losses)
    activations = np.array(activations)
    perfect_iters = np.array(perfect_iters)

    np.savez(f'data/xor_iters{ITERATIONS}_job{job_nr}.npz',
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
