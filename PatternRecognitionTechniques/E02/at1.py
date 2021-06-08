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
placa_1 = mpimg.imread('placa_1.jpg')
placa_2 = mpimg.imread('placa_2.jpg')
placas = mpimg.imread('placas.jpg')

# conversão para escala de cinza
# fonte: (https://www.kite.com/python/answers/how-to-convert-an-image-from-rgb-to-grayscale-in-python)
rgb_weights = [0.2989, 0.5870, 0.1140]
placa_1 = np.dot(placa_1[..., :3], rgb_weights)
# plt.imshow(placa_1, cmap=plt.get_cmap("gray"))
# plt.show()

placa_2 = np.dot(placa_2[..., :3], rgb_weights)
# plt.imshow(placa_2, cmap=plt.get_cmap("gray"))
# plt.show()

placas = np.dot(placas[..., :3], rgb_weights)
# plt.imshow(placas, cmap=plt.get_cmap("gray"))
# plt.show()

# initialize template match class
tm = TemplateMatcher()

# p=1 manhatam e p=2 euclidean
# [method, p, i, j] where i and j are subplot positions
configs = np.array([
    ['minkowski', 1, 0, 0],
                    ['minkowski', 2, 0, 1],
                    ['minkowski', 3, 0, 2],
                    ['minkowski', 0.5, 1, 0],
                    ['minkowski', 2.5, 1, 2],
                    ['heom', None, 1, 1],
                    ])

fig, ax = plt.subplots(nrows=2, ncols=3, subplot_kw={"projection": "3d"})
fig2, ax2 = plt.subplots()
ax2.imshow(placas)

for conf in configs:
    template = placa_1
    map = tm.match(placas, template, conf[0], conf[1])

    x, y = np.where(map == map.min())
    x = x[0]
    y = y[0]
    # Make data.
    X = np.arange(0, map.shape[1])
    Y = np.arange(0, map.shape[0])
    X, Y = np.meshgrid(X, Y)

    x_vec = np.arange(x, x+template.shape[1])
    y_vec = np.arange(y, y+template.shape[0])
    ax2.plot(x_vec, y*np.ones(x_vec.shape), 'r')
    ax2.plot(x*np.ones(y_vec.shape), y_vec, 'r')
    ax[int(conf[2]), int(conf[3])].plot_surface(X, Y, -map, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax[0, 0].set_title('minkowski - p=1')
ax[0, 1].set_title('minkowski - p=2')
ax[0, 2].set_title('minkowski - p=3')
ax[1, 0].set_title('minkowski - p=0.5')
ax[1, 1].set_title('minkowski - p=2.5')
ax[1, 2].set_title('heom')

# Show the plot.
# plt.show()
fig.show()
fig2.show()
print("end")
# x_mink_a, y_mink_a, map_mink_a = tm.match(placas, placa_1, 'minkowski', p=1)
# x_mink_b, y_mink_b, map_mink_b = tm.match(placas, placa_1, 'minkowski', p=2)
# x_mink_c, y_mink_c, map_mink_c = tm.match(placas, placa_1, 'minkowski', p=3)
# x_mink_d, y_mink_d, map_mink_d = tm.match(placas, placa_1, 'minkowski', p=0.5)
# x_heom, y_heom, map_heom = tm.match(placas, placa_1, 'heom', p=1)
# x_maha, y_maha, map_maha = tm.match(placas, placa_1, 'mahalanobis', p=1)

# Construct 2D histogram from data using the 'plasma' colormap
# Make data.
# X = np.arange(0, map_mink_a.shape[1])
# Y = np.arange(0, map_mink_a.shape[0])
# X, Y = np.meshgrid(X, Y)


# Plot the surface.
# fig, ax = plt.subplots(nrows=2, ncols=3, subplot_kw={"projection": "3d"})
# surf = ax[0, 0].plot_surface(X, Y, -map_mink_a, cmap=cm.coolwarm,
#                              linewidth=0, antialiased=False)
# # plt.title('minkowski (p=1)')
#
# surfb = ax[0, 1].plot_surface(X, Y, -map_mink_b, cmap=cm.coolwarm,
#                               linewidth=0, antialiased=False)
# # plt.title('minkowski (p=2)')
#
# surfc = ax[0, 2].plot_surface(X, Y, -map_mink_c, cmap=cm.coolwarm,
#                               linewidth=0, antialiased=False)
# # plt.title('minkowski (p=3)')
#
# ax[1, 1].plot_surface(X, Y, -map_mink_d, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)


# xs = np.ndarray([x_mink_a : x_mink_a+placa_1.shape[0], x_mink_b, x_mink_c, x_mink_d, x_heom, x_maha])
# plt.imshow(placas)
# # x = np.ndarray([x_mink_a: x_mink_a + placa_1.shape[0]])
# # y = np.ndarray([1,2])
# plt.plot(x, y)
# plt.show()
