import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


def get_distance(img1, img2, distance_metric='minkowski', p=1):
    d = np.inf
    if distance_metric == 'xor':
        d = xor(img1, img2)
    elif distance_metric == 'minkowski':
        d = minkowski(img1, img2, p)
    elif distance_metric == 'heom':
        d = heom(img1, img2)
    elif distance_metric == 'mahalanobis':
        d = mahalanobis(img1, img2)
    else:
        d = minkowski(img1, img2, p)
    return d


def minkowski(img1, img2, p):
    return pow(sum(sum(pow(abs(img1-img2), p))), 1/p)


def heom(img1, img2):
    return sum(sum(pow(img1*img2, 2)))

def xor(img1, img2):
    return sum(sum(img1^img2))

def mahalanobis(img1, img2):
    data = np.array([img1.flatten(), img2.flatten()])
    cov = np.cov(data)
    return distance.mahalanobis(img1.flatten(), img2.flatten(), np.linalg.inv(cov))


class TemplateMatcher(object):
    """
    Um perceptron simples com componentes simples.
    """

    def __init__(self):
        self.a = 1

    def match(self, image, template, distance_metric='minkowski', p=1):  # retorna melhor posicao

        max_x = image.shape[0] - template.shape[0]
        max_y = image.shape[1] - template.shape[1]
        d_matrix = np.zeros([max_y, max_x])

        # iterate on image
        for x in range(max_x):
            for y in range(max_y):
                cut_img = image[x: x + template.shape[0], y: y + template.shape[1]]
                d = get_distance(cut_img, template, distance_metric=distance_metric, p=p)
                d_matrix[y, x] = d

        return d_matrix
