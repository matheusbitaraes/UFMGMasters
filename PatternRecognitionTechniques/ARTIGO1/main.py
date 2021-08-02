from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from E03.knn import Knn
import pandas as pd
from matplotlib import cm
from sklearn import preprocessing
from scipy.optimize import differential_evolution, Bounds


# carregar 5 conjuntos de dados e transformá-los para binários (colocando saídas como -1 e 1).
data_sets = []

# WINE DATASET
wine = datasets.load_wine()
# filtra apenas dois tipos de vinho, para que fique binario
wine.data = wine.data[0:130, :]
wine.target = wine.target[0:130]
wine.target[wine.target == 0] = -1  # transformando y=0 em y=-1
wine.name = "Wine Dataset"
data_sets.append(wine)

# BREAST CANCER DATASET
bc = datasets.load_breast_cancer()
bc.target[bc.target == 0] = -1  # transformando y=0 em y=-1
bc.name = "Breast Cancer Dataset"
data_sets.append(bc)

# IRIS DATASET
iris = datasets.load_iris()
# juntando a classe 0 com a classe 1
iris.target[0:100] = 1
iris.target[100:] = -1

iris.name = "Iris Dataset"
data_sets.append(iris)


# CAESARIAN DATASET
caesarian = pd.read_csv('caesarian.csv')
caesarian.y[caesarian.y == 0] = -1
caesarian.target = caesarian.y.to_numpy()
caesarian.data = caesarian.to_numpy()
caesarian.data = caesarian.data[:, 0:5]
caesarian.name = "Caesarian Dataset"
data_sets.append(caesarian)

# CERVICAL CANCER DATASET
cervcan = pd.read_csv('sobar-72.csv')
cervcan.ca_cervix[cervcan.ca_cervix == 0] = -1
cervcan.target = cervcan.ca_cervix.to_numpy()
cervcan.data = cervcan.to_numpy()
cervcan.data = cervcan.data[:, 0:19]
cervcan.name = "Cervical Cancer Dataset"
data_sets.append(cervcan)


def knn_acc(v,x,y,num_executions=5,multiplier=-1): #esse V é pra ficar certo na funcao de otimizacao
    # dividir entre teste e treinamento
    p = v[0]
    k = int(v[1])
    acc = np.zeros(num_executions)
    for n in range(0, num_executions):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        knn = Knn(x_train, y_train)
        y_calculated = knn.classify_many(x=x_test, k=k, p=p)
        correct = sum(y_calculated == y_test)
        acc[n] = correct/len(y_test)
    return multiplier * np.mean(acc)

def knn_acc_2(h,x,y,num_executions=5,multiplier=-1): #esse V é pra ficar certo na funcao de otimizacao
    # dividir entre teste e treinamento
    acc = np.zeros(num_executions)
    for n in range(0, num_executions):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        knn = Knn(x_train, y_train)
        y_calculated = knn.classify_many_new_func(x=x_test, h=h)
        correct = sum(y_calculated == y_test)
        acc[n] = correct/len(y_test)
    return multiplier * np.mean(acc)


def get_average_accs(x, y, ps, ks, num_executions=5):
    ac = np.zeros([len(ps), len(ks)])
    for i in range(0, len(ps)):
        for j in range(0, len(ks)):
            # v = np.array([ps[i], ks[j]])
            # v[0] = ps[i]
            # v[1] = ks[j]
            ac[i, j] = knn_acc(np.array([ps[i], ks[j]]), x, y, num_executions=num_executions, multiplier=1)
    return ac


def get_average_accs_2(x, y, hs, num_executions=5):
    ac = np.zeros(len(hs))
    for i in range(0, len(hs)):
        ac[i] = knn_acc_2(hs[i], x, y, num_executions=num_executions, multiplier=1)
    return ac


for data_set in data_sets:
    if True:
        break
    x = data_set.data
    y = data_set.target
    dataset_name = data_set.name
    ps = np.arange(0.5, 4, 0.5)
    ks = np.arange(1, int(len(y)*0.5), int(len(y)*0.05))  # 50% de amostras sendo o maior k
    hs = np.arange(0.5, 4, 0.5)
    print(f"\n\n***DATASET {dataset_name}***")

    # normalização dos dados
    x = preprocessing.normalize(x, axis=0)

    acc = get_average_accs(x, y, ps, ks, num_executions=10)

    # otimizar o número de vizinhos k do knn e o p da métrica (é esperado busca em grid dos dois parametros para as bases consideradas).
    y_, x_ = np.unravel_index(acc.argmax(), np.array(acc).shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(ks, ps)
    ax.plot_surface(X, Y, acc, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.scatter(ks[x_], ps[y_], acc.max())
    plt.title(f'Região de acurácia - dados:{dataset_name}')
    plt.xlabel('k')
    plt.ylabel('p')
    ax.set_zlabel('Acurácia')

    # plt.show()
    # Utilizar um método de otimização para obter esses dados
    if x_ - 1 > 0 : kmin = ks[x_ - 1]
    else: kmin = ks[0]

    if x_ + 1 >= len(ks): kmax = ks[-1]
    else : kmax = ks[x_ + 1]

    if y_ - 1 > 0: pmin = ps[y_ - 1]
    else: pmin = ps[0]

    if y_ + 1 >= len(ps): pmax = ps[-1]
    else: pmax = ps[y_ + 1]
    print(f"bounds: \n k de {kmin} até {kmax} \n p de {pmin} até {pmax}")
    bounds = Bounds([pmin, kmin], [pmax, kmax])
    a = differential_evolution(knn_acc, args=(x, y), bounds=bounds, maxiter=100)
    print(f"Valores ótimos:\n p ótimo: {a.x[0]} \n k ótimo:{int(a.x[1])} \n acurácia média: { - a.fun}")
    # considerando-se uma certa regra de classificação, repetir os procedimentos anteriores
    # desafio: otimizar o parameto alpha



print("SEGUNDA PARTE")
for data_set in data_sets:
    x = data_set.data
    y = data_set.target
    dataset_name = data_set.name
    # ps = np.arange(0.5, 4, 0.5)
    # ks = np.arange(1, int(len(y)*0.5), int(len(y)*0.05))  # 50% de amostras sendo o maior k
    hs = np.arange(0.01, 2, 0.2)
    print(f"\n\n***DATASET {dataset_name}***")

    # normalização dos dados
    x = preprocessing.normalize(x, axis=0)

    acc = get_average_accs_2(x, y, hs, num_executions=10) # segunda parte do exercicio

    plt.plot(hs, acc)
    plt.title(f'Acurácia por h - dados:{dataset_name}')
    plt.xlabel('h')
    plt.ylabel('Acurácia')

    # otimizar o número de vizinhos k do knn e o p da métrica (é esperado busca em grid dos dois parametros para as bases consideradas).
    i_,  = np.unravel_index(acc.argmax(), np.array(acc).shape)
    if i_ - 1 > 0 : hmin = hs[i_ - 1]
    else: hmin = hs[0]

    if i_ + 1 >= len(hs): hmax = hs[-1]
    else : hmax = hs[i_ + 1]
    bounds = Bounds([hmin], [hmax])
    print(f"bounds: \n h de {hmin} até {hmax}")
    a = differential_evolution(knn_acc_2, args=(x, y), bounds=bounds, maxiter=200)
    print(f"Valores ótimos:\n h ótimo: {a.x[0]} \n acurácia média: { - a.fun}")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(ks, ps)
    # ax.plot_surface(X, Y, acc, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.scatter(ks[x_], ps[y_], acc.max())
    # plt.title(f'Região de acurácia - dados:{dataset_name}')
    # plt.xlabel('k')
    # plt.ylabel('p')
    # ax.set_zlabel('Acurácia')

    # plt.show()
    # Utilizar um método de otimização para obter esses dados
    # if x_ - 1 > 0 : kmin = ks[x_ - 1]
    # else: kmin = ks[0]
    #
    # if x_ + 1 >= len(ks): kmax = ks[-1]
    # else : kmax = ks[x_ + 1]
    #
    # if y_ - 1 > 0: pmin = ps[y_ - 1]
    # else: pmin = ps[0]
    #
    # if y_ + 1 >= len(ps): pmax = ps[-1]
    # else: pmax = ps[y_ + 1]
    # print(f"bounds: \n k de {kmin} até {kmax} \n p de {pmin} até {pmax}")
    # bounds = Bounds([pmin, kmin], [pmax, kmax])
    # a = differential_evolution(knn_acc, args=(x, y), bounds=bounds, maxiter=100)
    # print(f"Valores ótimos:\n p ótimo: {a.x[0]} \n k ótimo:{int(a.x[1])} \n acurácia média: { - a.fun}")
    # considerando-se uma certa regra de classificação, repetir os procedimentos anteriores
    # desafio: otimizar o parameto alpha
print("fim")