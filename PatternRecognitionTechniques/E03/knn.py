import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

def minkowsky(u, v, p):
    return pow(sum(pow(abs(u-v), p)), 1/p)


def distance(u, v, dist='minkowski', p=1):
    if dist == 'minkowski':
        return minkowsky(u, v, p)
    else:
        return None


class Knn(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_k_nearest(self, x, k, dist, p):  # retorna k primeiros y ordenados pela distancia
        d = np.zeros([self.x.shape[0], 2])
        for i in range(0, self.x.shape[0]):
            d[i, 0] = distance(x, self.x[i], dist, p)
            d[i, 1] = self.y[i]
        nearest = d[d[:, 0].argsort()]
        return nearest[1:k+1, 1]

    def get_all_neighbors(self, x, dist, p, h):  # retorna k primeiros y ordenados pela distancia
        d = np.zeros(self.x.shape[0])
        for i in range(0, self.x.shape[0]):
            d[i] = self.y[i] * self.k(x, self.x[i], h)
            # d[i, 1] = self.y[i]
        # nearest = d[d[:, 0].argsort()]
        return d

    def classify(self, xi, k):
        classifications = self.get_k_nearest(xi, k)
        return signal(sum(classifications))

    def classify_many(self, x, k, distance="minkowski", p=1):
        y = np.zeros(x.shape[0])
        for i in range(0, x.shape[0]):
            y[i] = signal(sum(self.get_k_nearest(x[i], k, distance, p)))
        return y

    def classify_many_new_func(self, x, h, distance="minkowski", p=1, alpha=1):
        y = np.zeros(x.shape[0])
        for i in range(0, x.shape[0]):
            # alpha(vetor) * yi * k(xi, xj)
            y[i] = signal(sum(alpha * self.get_all_neighbors(x[i], distance, p, h)))
        return y

    def k(self, xi, xj, h):
        return np.exp(-np.power(distance(xi, xj, 'minkowski', 2)/h, 2))

    def plot_region(self, x, y, k, x_min=0, x_max=8):
        # fig = plt.figure()
        # ax = fig.add_subplot()
        fig, ax = plt.subplots()
        x1 = np.arange(x_min, x_max + 1, .5)
        x2 = np.arange(x_min, x_max + 1, .5)
        z = np.zeros([x1.shape[0], x2.shape[0]])
        x1, x2 = np.meshgrid(x1, x2)
        for i in range(x_min, x1.shape[0]):
            for j in range(x_min, x2.shape[0]):
                z[i, j] = self.classify(np.array([x1[i, j], x2[i, j]]), k)
        # ax.plot_surface(x1, x2, z, cmap='seismic', linewidth=0, antialiased=False, alpha=0.3)
        c = ax.pcolormesh(x1, x2, z, cmap='seismic', shading='auto')
        # ax.scatter(x1, x2, z, color='gray')
        xa1 = x[np.where(y > 0), 0]
        xa2 = x[np.where(y > 0), 1]
        xb1 = x[np.where(y < 0), 0]
        xb2 = x[np.where(y < 0), 1]
        ax.plot(xa1, xa2, '*', color='red')
        ax.plot(xb1, xb2, '*', color='blue')
        ax.set_title(f"Região do KNN para k={k}")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        # ax.set_zlabel('y')
        ax.axis([x1.min(), x1.max(), x2.min(), x2.max()])
        # fig.colorbar(ax=ax)
        plt.show()

    def plot_data(self, x, y):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        xa1 = x[np.where(y > 0), 0]
        xa2 = x[np.where(y > 0), 1]
        xb1 = x[np.where(y < 0), 0]
        xb2 = x[np.where(y < 0), 1]
        ax.scatter(xa1, xa2, y[y > 0], color='red')
        ax.scatter(xb1, xb2, y[y < 0], color='blue')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        plt.show()


def signal(yi):
    if yi >= 0:
        return 1
    else:
        return -1
