import itertools
import math
from typing import List, Tuple, Callable
import sys

import numpy as np

import weight_sharing_and as ws

count = 0


def generate_renaming(weight_ls: List[int], combination: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    all_renamed = []
    renames = itertools.permutations(weight_ls)
    for rename in renames:
        renaming_dict = {i: j for i, j in zip(weight_ls, rename)}
        renamed = tuple([renaming_dict.get(number) for number in combination])
        # renamed = tuple(map(renaming_dict.get, combination))
        all_renamed.append(renamed)
    return all_renamed


def generate_resigning(weight_ls: List[int], combination: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    all_renamed = [combination]
    resigns = []
    for i in range(len(weight_ls)):
        to_resign = itertools.combinations(weight_ls, i+1)
        resigns += list(to_resign)

    for resign in resigns:
        base = [1 for _ in range(len(combination))]
        for number in resign:
            base = [-1*el if abs(combination[i]) == number else el for i, el in enumerate(base)]
        comb_arr = np.array(base) * np.array(combination)
        all_renamed.append(tuple(comb_arr.tolist()))
    return all_renamed


def filter_combinations(weight_ls: List[int], combinations: List[Tuple[int, ...]],
                        generator: Callable[[List[int], Tuple[int, ...]], List[Tuple[int, ...]]]
                        ) -> List[Tuple[int, ...]]:
    ok_combs = []
    for combination in combinations:
        all_renamings = generator(weight_ls, combination)
        if any([renaming in ok_combs for renaming in all_renamings]):
            continue
        ok_combs.append(combination)
    return ok_combs


def first_level_combinations() -> List[Tuple[int, ...]]:
    all_combs = []
    for NR_OF_INDIV_WEIGHTS in range(1, 2):
        weight_ls = list(range(1, NR_OF_INDIV_WEIGHTS+1))
        combins = itertools.product(weight_ls, repeat=2)
        combins = list(filter(lambda x: len(set(x)) == NR_OF_INDIV_WEIGHTS, combins))
        combins = filter_combinations(weight_ls, combins, generate_renaming)
        combins = second_level_combinations(weight_ls, combins)
        all_combs += combins
    return all_combs


def second_level_combinations(weight_ls: List[int], combins: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    ones_plus_minus = list(itertools.product([1, -1], repeat=2))
    plus_minus_combs = []
    for combination in combins:
        for plus_minus in ones_plus_minus:
            res = np.array(plus_minus) * np.array(combination)
            plus_minus_combs.append(tuple(res.tolist()))
    combins = plus_minus_combs
    combins = filter_combinations(weight_ls, combins, generate_resigning)
    return combins


if __name__ == "__main__":
    ITERATIONS = 2000
    combs: List[Tuple[int, ...]]
    combs = first_level_combinations()
    print(combs)

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
        print(f'OUTER {i + 1}/{len(combs)}')
        weight_list = list(set(map(abs, comb)))
        inits = list(itertools.product([-0.1, -0.01, 0.01, 0.1], repeat=len(weight_list)))
        print()
        for j, initial in enumerate(inits):
            init_dict = dict(zip(weight_list, initial))
            print(f'Inner {i + 1}.: {j + 1}/{len(inits)}')
            miscl, cl, w, b, l, ac, pi = ws.train(comb, init_dict, ITERATIONS)
            combinations.append(combs)
            initializations.append(initial)
            misclassified.append(miscl)
            classes.append(cl)
            weights.append(w)
            biases.append(b)
            losses.append(l)
            activations.append(ac)
            perfect_iters.append(pi)
            print()
        print(20*'-' + '\n\n')

    combinations = np.array(combinations)
    initializations = np.array(initializations)
    misclassified = np.array(misclassified)
    classes = np.array(classes)
    weights = np.array(weights)
    biases = np.array(biases)
    losses = np.array(losses)
    activations = np.array(activations)
    perfect_iters = np.array(perfect_iters)

    np.savez(f'and_iters{ITERATIONS}.npz',
             comb=combinations,
             initial=initializations,
             miscl=misclassified,
             cl=classes,
             w=weights,
             b=biases,
             l=losses,
             act=activations,
             pi=perfect_iters)
    # np.savez('weight_sharing_and.npz', miscl=misclassified, cl=classes, w=weights, l=losses, act=activations)
