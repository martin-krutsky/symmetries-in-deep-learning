from typing import Dict, List

import _dynet as dy
dyparams = dy.DynetParams()
dyparams.set_random_seed(0)
dyparams.init()

import numpy as np


data = [([0, 0, 0], 0),
        ([0, 0, 1], 1),
        ([0, 1, 0], 1),
        ([1, 0, 0], 1),
        ([1, 0, 1], 0),
        ([1, 1, 0], 0),
        ([0, 1, 1], 0),
        ([1, 1, 1], 1)]


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

    b1 = model.parameters_from_numpy(np.array([0, 0, 0, 0]))
    b2 = model.parameters_from_numpy(np.array([0]))

    x = dy.vecInput(3)
    y = dy.scalarInput(0)
    h11 = dy.logistic((coeffs[0]*weights[0]*x[0])+(coeffs[1]*weights[1]*x[1])+(coeffs[2]*weights[2]*x[2]) + b1[0])
    h12 = dy.logistic((coeffs[3]*weights[3]*x[0])+(coeffs[4]*weights[4]*x[1])+(coeffs[5]*weights[5]*x[2]) + b1[1])
    h13 = dy.logistic((coeffs[6]*weights[6]*x[0])+(coeffs[7]*weights[7]*x[1])+(coeffs[8]*weights[8]*x[2]) + b1[2])
    h14 = dy.logistic((coeffs[9]*weights[9]*x[0])+(coeffs[10]*weights[10]*x[1])+(coeffs[11]*weights[11]*x[2]) + b1[3])
    y_pred = dy.logistic((coeffs[12]*weights[12]*h11)+(coeffs[13]*weights[13]*h12)+(coeffs[14]*weights[14]*h13)+(coeffs[15]*weights[15]*h14) + b2)

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
            x3 = (mi // 4) % 2
            x.set([T if x1 else F, T if x2 else F, T if x3 else F])
            y.set(T if (x1+x2+x3) % 2 else F)

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
    bias_ls = [b1.value(), b2.value()]
    return misclass, predicted, weight_ls, bias_ls, mloss, activations, perfect_iteration


