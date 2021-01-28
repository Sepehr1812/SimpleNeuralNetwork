""" neural network algorithm implementation """

from math import exp
from random import normalvariate, uniform
from re import split

import matplotlib.pyplot as plt
import numpy as np


def read_data():
    """
    read from a dataset csv file and returns dataset as an array of tuples
    """
    dataset = []
    with open("dataset.csv", "r") as f:
        data: list[list] = [split(",", line.rstrip('\n')) for line in f]
        data.pop(0)

        for i in range(len(data)):
            data[i] = [float(x) for x in data[i]]
            data[i][2] = int(data[i][2])

        for d in data:
            dataset.append(tuple(d))

    return dataset


def plot(dataset: list):
    """
    plotting dataset
    """
    class0 = []
    class1 = []

    for d in dataset:
        if d[2] == 0:
            class0.append(d)
        else:
            class1.append(d)

    plt.scatter([x[0] for x in class0], [x[1] for x in class0], label="0", color="blue", marker="o", s=10)
    plt.scatter([x[0] for x in class1], [x[1] for x in class1], label="1", color="red", marker="*", s=20)

    plt.legend()
    plt.show()


def sigmoid(x: float):
    return 1 / (1 + exp(-x))


def compute_y(w: list, x: list):
    return np.dot(x, w[1:]) + w[0]


def compute_cost(w: list, x: list, y0: int, m: int, num: int):
    """
    :param w: weights
    :param x: coordinates of data
    :param y0: label of data
    :param m: m in cost formula
    :param num: number of weight
    :return: dcost/dw
    """
    y = sigmoid(compute_y(w, x))

    if not num == 0:
        return (2 / m) * x[num - 1] * y * (1 - y) * (y - y0)
    return (2 / m) * y * (1 - y) * (y - y0)


def train(dataset: list, alpha: float):
    """
    trains data
    :param dataset: dataset the function is working on
    :param alpha: learning ratio
    :return: list of weight
    """
    w = []  # zero index represents b
    grad = [0, 0, 0]

    w.append(normalvariate(0, 1))
    w.append(normalvariate(0, 1))
    w.append(normalvariate(0, 1))

    for i in range(3000):
        for k in range(3):
            grad[k] = 0
            for data in dataset:
                grad[k] += compute_cost(w, [data[0], data[1]], data[2], len(dataset), k)

        _sum = 0  # to check convergence
        for j in range(3):
            _sum += abs(alpha * grad[j])
            w[j] -= alpha * grad[j]

        if abs(_sum) < 0.00003:  # to check convergence
            return w

    return w


def test(w: list, dataset: list):
    """
    tests some data
    :param w: array of weights
    :return: tested data, precision
    """
    new_dataset = []
    p = 0

    for data in dataset:
        sigmoid_y = sigmoid(compute_y(w, [data[0], data[1]]))
        if sigmoid_y >= 0.5:
            if data[2] == 1:
                p += 1
            new_dataset.append((data[0], data[1], 1))
        else:
            if data[2] == 0:
                p += 1
            new_dataset.append((data[0], data[1], 0))

    return new_dataset, p / len(dataset) * 100


def main():
    alpha = 1.5
    dataset = read_data()
    # plot(dataset)

    train_dataset = []
    test_dataset = []

    # setting train and test datasets
    for d in dataset:
        if uniform(0, 1) > 0.75:
            test_dataset.append(d)
        else:
            train_dataset.append(d)

    w = train(train_dataset, alpha)
    dataset_test, precision = test(w, test_dataset)

    print("precision:", precision)
    plot(dataset_test)


if __name__ == '__main__':
    main()
