import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib.image as mpimg
from TemplateMatcher import TemplateMatcher
import plotly.graph_objects as go

from mlxtend.plotting import plot_decision_regions

# se nao conseguir na mão, usar o opencv
# https://towardsdatascience.com/object-detection-on-python-using-template-matching-ab4243a0ca62
# leitura dos dados
_0 = mpimg.imread('0.png')
_3 = mpimg.imread('3.png')
_5 = mpimg.imread('5.png')
_7 = mpimg.imread('7.png')
_8 = mpimg.imread('8.png')
_d = mpimg.imread('D.png')
_f = mpimg.imread('F.png')
_j = mpimg.imread('J.png')
_u = mpimg.imread('U.png')
_x = mpimg.imread('X.png')
p1 = mpimg.imread('JFD730.png')
p2 = mpimg.imread('JUX580.png')
placas = mpimg.imread('placas.jpg')

# conversão para escala de cinza
# fonte: (https://www.kite.com/python/answers/how-to-convert-an-image-from-rgb-to-grayscale-in-python)
rgb_weights = [0.2989, 0.5870, 0.1140]

placas = np.dot(placas[..., :3], rgb_weights)
threshold = 137
_0 = np.dot(_0[..., :3], rgb_weights)
_3 = np.dot(_3[..., :3], rgb_weights)
_5 = np.dot(_5[..., :3], rgb_weights)
_7 = np.dot(_7[..., :3], rgb_weights)
_8 = np.dot(_8[..., :3], rgb_weights)
_d = np.dot(_d[..., :3], rgb_weights)
_f = np.dot(_f[..., :3], rgb_weights)
_j = np.dot(_j[..., :3], rgb_weights)
_u = np.dot(_u[..., :3], rgb_weights)
_x = np.dot(_x[..., :3], rgb_weights)
p1 = np.dot(p1[..., :3], rgb_weights)
p2 = np.dot(p2[..., :3], rgb_weights)

#binarize
placas = np.where(placas > threshold, 1, -1)
_0 = np.where(_0 > 0.5, 1, -1)
_3 = np.where(_3 > 0.5, 1, -1)
_5 = np.where(_5 > 0.5, 1, -1)
_7 = np.where(_7 > 0.5, 1, -1)
_8 = np.where(_8 > 0.5, 1, -1)
_f = np.where(_f > 0.5, 1, -1)
_j = np.where(_j > 0.5, 1, -1)
_u = np.where(_u > 0.5, 1, -1)
_d = np.where(_d > 0.5, 1, -1)
_x = np.where(_x > 0.5, 1, -1)
p1 = np.where(p1 > 0.5, 1, -1)
p2 = np.where(p2 > 0.5, 1, -1)

# initialize template match class
tm = TemplateMatcher()
# p=1 manhatam e p=2 euclidean
# [method, p, i, j] where i and j are subplot positions

fig, ax = plt.subplots()
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax.imshow(placas, cmap=plt.get_cmap("gray"))
# fig2, ax2 = plt.subplots()
# ax2.imshow(_j, cmap=plt.get_cmap("gray"))
# 40 de altura
chars = []
chars.append(_j)
chars.append(_f)
chars.append(_d)
chars.append(_7)
chars.append(_3)
chars.append(_0)
# chars.append(p1)
# chars.append(p2)
#JFD-703
#//.arange([_j, _u, _x, _5, _8, _0])
matches = np.zeros((2, len(chars)))
for char in chars:
    print('next char ...')
    template = char
    map = tm.match(placas, template, 'xor')

    x, y = np.where(map == map.max())

    x_vec = np.arange(x, x+template.shape[1])
    y_vec = np.arange(y, y+template.shape[0])
    ax.plot(x_vec, y*np.ones(x_vec.shape), 'r')
    ax.plot(x*np.ones(y_vec.shape), y_vec, 'r')

    X = np.arange(0, map.shape[1])
    Y = np.arange(0, map.shape[0])
    X, Y = np.meshgrid(X, Y)
    ax2.plot_surface(X, Y, map, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    print('processed')

# X = np.arange(0, map.shape[1])
# Y = np.arange(0, map.shape[0])
# X, Y = np.meshgrid(X, Y)
# fig_map, ax_map = plt
# ax_map.plot_surface(X, Y, -sum_map, cmap=cm.coolwarm, linewidth=0, antialiased=False)

print("end")
