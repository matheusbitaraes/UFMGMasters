import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# gerar amostras normals centradas no 2,2 e 4,4
muA = 2
sigmaA = 0.4
muB = 4
sigmaB = 0.4
x1a = np.random.normal(muA, sigmaA, 50)
x2a = np.random.normal(muA, sigmaA, 50)
x1b = np.random.normal(muB, sigmaB, 50)
x2b = np.random.normal(muB, sigmaB, 50)

# plotar amostras
fig = plt.figure()
plt.plot(x1a, x2a, '*b')
plt.plot(x1b, x2b, '*r')
plt.title(f'Amostragem de classes A e B')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# calcular densidades
def dens(x1, x2, mu, sigma):
    e = np.exp(-0.5 * (np.power((x1 - mu) / sigma, 2) + np.power((x2 - mu / sigma), 2)))
    return e / (2 * np.pi * sigma * sigma)


def p(x, mu, sigma):
    mu = np.array([mu, mu])
    cov_mat = np.array([[sigma, 0], [0, sigma]])
    e = np.exp(-0.5 * np.transpose(x - mu) @ np.linalg.inv(cov_mat) @ (x - mu))
    deno = np.sqrt(np.power(2 * np.pi, 2) * np.linalg.det(cov_mat))
    bayes = e/deno
    return bayes
    # if bayes[1] > bayes[0]:
    #     return 1
    # else:
    #     return -1


# plotar densidades 2d e 3d
x1vec = np.arange(0, 7, 0.1)
x2vec = np.arange(0, 7, 0.1)
den = np.zeros([len(x1vec), len(x2vec)])
for i in range(0, len(x1vec)):
    for j in range(0, len(x2vec)):
        den[i, j] = dens(x1vec[i], x2vec[j], muA, np.sqrt(sigmaA)) + dens(x1vec[i], x2vec[j], muB, np.sqrt(sigmaB))
# den = dens(x1a, x2a)

X1, X2 = np.meshgrid(x1vec, x2vec)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, den, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel('x1')
plt.ylabel('x2')
ax.set_zlabel('Densidade de probabilidade')
plt.title('Densidade de probabilidade das classes A e B')
plt.show()

grid = np.zeros([len(x1vec), len(x2vec)])
dens2 = np.zeros([len(x1vec), len(x2vec)])
for i in range(0, len(x1vec)):
    for j in range(0, len(x2vec)):
        pA = p(np.array([x1vec[i], x2vec[j]]), muA, sigmaA)
        pB = p(np.array([x1vec[i], x2vec[j]]), muB, sigmaB)
        dens2[i, j] = pA + pB
        if pA > pB:
            grid[i, j] = -1
        else:
            grid[i, j] = 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, dens2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel('x1')
plt.ylabel('x2')
ax.set_zlabel('Densidade de probabilidade')
plt.title('Densidade de probabilidade das classes A e B (parte 2)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel('x1')
plt.ylabel('x2')
ax.set_zlabel('Classificação')
plt.title('Superfície de separação')
plt.show()
