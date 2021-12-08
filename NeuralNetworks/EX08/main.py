# O objetivo do exercício desta semana é combinar os conceitos aprendidos na Unidade 2 e cons-
# truir uma rede neural que soma elementos das redes RBF e das redes ELM.
# As bases de dados a serem estudadas são as mesmas do exercício 6:
# • Breast Cancer (diagnostic)
# • Statlog (Heart)
# Os mesmos cuidados para separação de conjunto de treinamento e teste, já mencionados no
# enunciado do exercício 6, devem ser tomados, bem como deve ser dada atenção ao escalonamento
# dos dados (entre [0; 1] ou [−1; 1]).
# Para o exercício desta semana, o aluno deve combinar os algoritmos de treinamento de redes
# ELM e RBF: construir uma rede RBF com centros e raios atribuídos de forma aleatória
# aos neurônios. Uma possibilidade, que não é a única nem a melhor, para a construção de
# centros é colocá-los entre 2 pontos escolhidos aleatoriamente do conjunto de treinamento, com
# o raio da função igual à distância entre os pontos.
# Além da RBF com centros e raios aleatórios, deve ser construída uma RBF com centros e raios
# selecionados a partir do k-médias. As acurácias obtidas por cada uma das redes nas duas bases
# devem ser apresentadas no formato 𝑚𝑒𝑑𝑖𝑎 ± 𝑑𝑒𝑠𝑣𝑖𝑜 e comparadas com os resultados obtidos no
# exercício 6 para ELMs.
# Deve ser comparado, também, o número de centros necessários para desempenho semelhante
# entre as redes RBF com centros aleatórios e com centros selecionados por agrupamento (k-
# médias).
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris

from ELM import ELM
from RBF import RBF
from plot_utils import plot_result
from utils import multiple_executions, print_mean_std

NUM_EXECUTIONS = 50
dataset_list = []

# ##### IRIS DATASET ######
# ds = load_iris()
# x = ds.data
# y = ds.target
# x = x[y != 2]
# y = y[y != 2]
# y[y == 0] = -1
#
# dataset_list.append({"name": "Iris Dataset",
#                      "x": x,
#                      "y": y,
#                      "model_list": [
#                          RBF({
#                              "name": "RBF",
#                              "use_kmeans": True,
#                              "p": 2
#                          }),
#                      ],
#                      })

##### BREAST CANCER DATASET ######
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
y[y == 0] = -1

dataset_list.append({"name": "Breast Cancer",
                     "x": x,
                     "y": y,
                     "model_list": [
                         # ELM({
                         #     "name": "ELM",
                         #     "p": 15
                         # }),
                         # RBF({
                         #     "name": "RBF sem Kmeans",
                         #     "use_kmeans": False,
                         #     "p": 2
                         # }),
                         RBF({
                             "name": "RBF com Kmeans",
                             "use_kmeans": True,
                             "p": 2
                         })
                     ]
                     })

##### HEART DATASET ######
df = pd.read_csv('heart.dat', sep=" ", header=None)
dataset = df.to_numpy()
n_col = dataset.shape[1]
x = dataset[:, 0:n_col - 1]
y = dataset[:, n_col - 1]
y[y == 2] = -1

dataset_list.append({"name": "Heart Dataset",
                     "x": x,
                     "y": y,
                     "model_list": [
                         # ELM({
                         #     "name": "ELM",
                         #     "p": 15
                         # }),
                         # RBF({
                         #     "name": "RBF sem Kmeans",
                         #     "use_kmeans": False,
                         #     "p": 2
                         # }),
                         RBF({
                             "name": "RBF com Kmeans",
                             "use_kmeans": True,
                             "p": 2
                         })
                     ]
                     })

for dataset in dataset_list:
    model_list = dataset["model_list"]
    name = dataset["name"]
    accuracy_lists = []
    labels = []
    for i in range(0, len(model_list)):
        accuracy_list = multiple_executions(X=x, Y=y, model=model_list[i], num_executions=NUM_EXECUTIONS)
        accuracy_lists.append(accuracy_list)
        labels.append(model_list[i].name)
        print_mean_std(accuracy_list, model_list[i], name)
        plot_result(accuracy_list, labels, title=f"Acurácias para dataset {name}")


