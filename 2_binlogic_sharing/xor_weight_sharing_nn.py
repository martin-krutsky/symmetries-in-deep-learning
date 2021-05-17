import dynet as dy
import numpy as np
import random

data = [([0, 1], 1),
        ([1, 1], 1),
        ([0, 0], 0),
        ([0, 1], 0)]


def cross_entropy(y, yhat):
    if y == 0:
        loss = -dy.log(1 - yhat)
    elif y == 1:
        loss = -dy.log(yhat)
    return loss


def train():
    model = dy.ParameterCollection()
    trainer = dy.SimpleSGDTrainer(model)

    W1 = model.parameters_from_numpy(np.array([[1], [1]]))
    b1 = model.parameters_from_numpy(np.array([0, 0]))
    W2 = model.parameters_from_numpy(np.array([[1]]))
    b2 = model.parameters_from_numpy(np.array([0]))

    x = dy.vecInput(2)
    y = dy.scalarInput(0)
    h = dy.logistic((W1*x[0])+(W1*x[1]) + b1)

    y_pred = dy.logistic((-W2*h[0])+(W2*h[1]) + b2)
    loss = dy.binary_log_loss(y_pred, y)
    T = 1
    F = 0

    for iter in range(140):
        mloss = 0.0
        misclass = 0
        for mi in range(4):
            x1 = mi % 2
            x2 = (mi // 2) % 2
            x.set([T if x1 else F, T if x2 else F])
            y.set(T if x1 != x2 else F)

            pred = 1 if y_pred.value() > 0.5 else 0
            print(pred, y.value(), y_pred.value())
            # print(loss.scalar_value())
            if pred != int(y.value()):
                misclass += 1

            mloss += loss.scalar_value()
            loss.backward()
            trainer.update()
        mloss /= 4.
        print("loss: %0.9f" % mloss)
        print(f'{misclass}/4 Missclassified')

    # x.set([F, T])
    # z = -(-y_pred)
    # print(z.scalar_value())

    # model = dy.ParameterCollection()
    # # learnable_subcol = model.add_subcollection('learnable')
    # # nonlearnable_subcol = model.add_subcollection('nonlearnable')
    #
    # W1_1 = model.parameters_from_numpy(np.array([[1], [1]]))
    # # W1_2 = nonlearnable_subcol.parameters_from_numpy(np.array([[-1], [-1]]))
    # # W1_2.set_updated(False)
    # b1 = model.parameters_from_numpy(np.array([0, 0]))
    # W2_1 = model.parameters_from_numpy(np.array([[1]]))
    # # W2_2 = nonlearnable_subcol.parameters_from_numpy(np.array([[-1]]))
    # # W2_2.set_updated(False)
    # b2 = model.parameters_from_numpy(np.array([0]))
    #
    # # W1 = model.parameters_from_numpy(np.array([[1, 1], [1, 1]]))
    # # b1 = model.parameters_from_numpy(np.array([0, 0]))
    # # W2 = model.parameters_from_numpy(np.array([[1, 1]]))
    # # b2 = model.parameters_from_numpy(np.array([0]))
    #
    # # print(learnable_subcol.parameters_list())
    # # print(nonlearnable_subcol.parameters_list())
    #
    # trainer = dy.SimpleSGDTrainer(model)
    #
    # for i in range(5000):
    #     closs = 0.0
    #     # random.shuffle(data)
    #     misclass = 0
    #     for x, y in data:
    #         # dy.renew_cg()
    #         x = dy.inputVector(x)
    #
    #         y1 = dy.logistic(np.dot(W1_1, x[0]) + np.dot(-W1_1, x[1]) + b1)
    #         yhat = dy.logistic(np.dot(W2_1, y1[0]) + np.dot(-W2_1, y1[1]) + b2)
    #
    #         pred = 1 if yhat.value() > 0.5 else 0
    #         if pred != y:
    #             misclass += 1
    #
    #         loss = cross_entropy(y, yhat)
    #         closs += loss.scalar_value()
    #         loss.backward()
    #         trainer.update()
    #
    #         # W1_2.set_value(-1*W1_1.value())
    #         # print(W2_1.value())
    #         # print(W2_2.value())
    #         # W2_2.set_value(-1*W2_1.value())
    #         # print(W2_2.is_updated())
    #         # print(W2_2.value())
    #
    #     if (i + 1) % 100 == 0:
    #         print(f"Iteration: {i+1}, Loss: {closs}, Misclass: {misclass}")
    #
    # print(f'W1 {W1_1.value()}')
    # # print(f'W1 {W1_2.value()}')
    # print(f'b1 {b1.value()}')
    # print(f'W2 {W2_1.value()}')
    # # print(f'W2 {W2_2.value()}')
    # print(f'b2 {b2.value()}')


if __name__ == "__main__":
    train()

