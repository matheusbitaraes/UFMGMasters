# ELM model

import numpy as np
from numpy.linalg import linalg
from sklearn.cluster import KMeans

class RBF:
    def __init__(self, attributes):
        self.name = attributes["name"]
        self.p = attributes["p"]
        self.should_use_kmeans = attributes["use_kmeans"]
        self.W = None
        self.H = None
        self.covlist = []
        self.m = None

    def get_name(self):
        return self.name

    def pdf_n_var(self, x, m, K, n):
        if n == 1:
            r = np.sqrt(K)
            px = (1/(np.sqrt(2*np.pi()*r*r)))*np.exp(-0.5 * pow((x-m)/r, 2))
        else:
            det = np.linalg.det(K)
            if det == 0:
                det = 2e-18
            part_a = 1 / (np.sqrt(pow(2 * np.pi, n)) * det)
            term = np.matmul(np.matrix.transpose(x-m), np.linalg.pinv(K))
            part_b = np.exp(-.5 * (np.matmul(term, (x-m))))
            px = part_a * part_b

        return px

    def train(self, data):
        n_rows, n_cols = data.shape
        n = n_cols - 1
        x = data[:, 0:n]
        y = data[:, n]

        # a = np.matrix([[1, 2, 3, 4, 6],[5, 6, 7, 8, 6], [9, 10, 11, 12, 6]])
        # np.cov(a)
        if self.should_use_kmeans:
            kmeans = KMeans(self.p).fit(x)
            self.m = kmeans.cluster_centers_
        else:
            self.m = []
            kmeans = {
                "labels_": [1, 2, 3]
            }

        covariance_matrices = self.estimate_covariance_matrices(x, kmeans)

        H = np.zeros([n_rows, self.p])
        for j in range(0, n_rows):
            for i in range(0, self.p):
                cov_matrix = covariance_matrices[i]
                H[j, i] = self.pdf_n_var(x[j, ], self.m[i, :], cov_matrix, n_cols)

        Haug = np.c_[np.ones(n_rows), H]

        termA = np.matmul(np.transpose(Haug), Haug)
        matA = np.matmul(np.linalg.pinv(termA), np.transpose(Haug))
        self.W = np.matmul(matA, y)
        self.covlist = covariance_matrices
        self.H = H

        return None

    def estimate_covariance_matrices(self, x, xclust):
        covariance_list = []
        for i in range(0, self.p):
            xci = x[xclust.labels_ == i]
            covariance_matrix = np.cov(np.transpose(xci))
            covariance_list.append(covariance_matrix)

        return covariance_list

    def get_accuracy_and_error(self, data):
        n_rows, n_cols = data.shape
        x = data[:, 0:n_cols - 1]
        y = data[:, n_cols - 1]

        y_pred = self.eval(x)

        acc = len(y[y_pred == y]) / len(y)
        return acc

    def eval(self, x):
        n_rows, n_cols = x.shape
        H = np.zeros([n_rows, self.p])
        for j in range(0, n_rows):
            for i in range(0, self.p):
                mi = self.m[i, :]
                covi = self.covlist[i]
                H[j, i] = self.pdf_n_var(x[j, :], mi, covi, n_cols)

        Haug = np.c_[np.ones(n_rows), H]
        Yhat = np.matmul(Haug, self.W)
        return np.sign(Yhat)
