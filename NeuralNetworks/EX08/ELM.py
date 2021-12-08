# ELM model
import math

import numpy as np
from numpy.linalg import linalg


class ELM:
    def __init__(self, attributes):
        self.name = attributes["name"]
        self.p = attributes["p"]
        self.Z = None
        self.W = None

    def get_name(self):
        return self.name

    def train(self, data):
        n_rows, n_cols = data.shape
        x = data[:, 0:n_cols - 1]
        y = data[:, n_cols - 1]

        xin = np.c_[np.ones(n_rows), x]
        self.Z = np.random.random([n_cols, self.p]) - 0.5

        H = np.tanh(np.matmul(xin, self.Z))
        self.W = np.matmul(linalg.pinv(H), y)

        y_hat_train = np.sign(np.matmul(H, self.W))
        train_err = sum(pow(y - y_hat_train, 2)) / 4

        return train_err

    def get_accuracy_and_error(self, data):
        n_rows, n_cols = data.shape
        x = data[:, 0:n_cols - 1]
        y = data[:, n_cols - 1]

        y_pred = self.eval(x)

        acc = len(y[y_pred == y]) / len(y)
        return acc

    def eval(self, x):
        n_rows, n_cols = x.shape
        x = np.c_[np.ones(n_rows), x]
        # self.Z = np.random.random([n_cols + 1, self.p]) - 1
        H = np.tanh(np.matmul(x, self.Z))
        pred = np.sign(np.matmul(H, self.W))

        return pred
