from typing import Dict, List

import _dynet as dy
dyparams = dy.DynetParams()
dyparams.set_random_seed(0)
dyparams.init()

import numpy as np


data = [([0, 1], 1),
        ([1, 1], 1),
        ([0, 0], 0),
        ([0, 1], 1)]


def cross_entropy(y, yhat):
    if y == 0:
        loss = -dy.log(1 - yhat)
    elif y == 1:
        loss = -dy.log(yhat)
    return loss


def train(combination: List[int], init_dict: Dict[int, int], iters: int):
    model = dy.ParameterCollection()
    trainer = dy.SimpleSGDTrainer(model)

    used_numbers = {}
    weights = [None for _ in range(len(combination))]
    coeffs = np.sign(combination)
    for i, number in enumerate(combination):
        abs_number = abs(number)
        if abs_number in used_numbers:
            weights[i] = weights[used_numbers[abs_number]]
        else:
            used_numbers[abs_number] = i
            # weights[i] = model.parameters_from_numpy(np.array([1]))
            weights[i] = model.parameters_from_numpy(np.array([init_dict[abs_number]]))

    b1 = model.parameters_from_numpy(np.array([0]))

    x = dy.vecInput(2)
    y = dy.scalarInput(0)

    y_pred = dy.logistic((coeffs[0]*weights[0]*x[0])+(coeffs[1]*weights[1]*x[1]) + b1)
    loss = dy.binary_log_loss(y_pred, y)
    T = 1
    F = 0
    predicted = []
    activations = []
    misclass = 0
    mloss = 0.0
    perfect_iteration = None

    for iter in range(iters):
        mloss = 0.0
        misclass = 0
        for mi in range(4):
            x1 = mi % 2
            x2 = (mi // 2) % 2
            x.set([T if x1 else F, T if x2 else F])
            y.set(T if x1 or x2 else F)

            pred = 1 if y_pred.value() > 0.5 else 0
            if (iter + 1) % iters == 0:
                predicted.append(pred)
                activations.append(y_pred.value())
                # print(pred, y.value(), y_pred.value())

            # print(loss.scalar_value())
            if pred != int(y.value()):
                misclass += 1

            mloss += loss.scalar_value()
            loss.backward()
            trainer.update()
        mloss /= 4.
        if perfect_iteration is None and misclass == 0:
            perfect_iteration = iter+1
        if (iter + 1) % iters == 0:
            weight_ls = [w.value() for w in weights]
            # print(weight_ls)
            # print('iteration:', iter+1)
            # print("loss: %0.9f" % mloss)
            print(f'{misclass}/4 Missclassified')
            print('perfect_iteration', perfect_iteration)

    weight_ls = [w.value() for w in weights]
    bias_ls = [b1.value()]
    return misclass, predicted, weight_ls, bias_ls, mloss, activations, perfect_iteration


