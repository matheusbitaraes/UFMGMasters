# KNN exercise
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from knn import Knn




# create two dimensional data
data_simple = np.array([[1, 0.5],
                        [1, 0.8],
                        [2, 1.7],
                        [2, 1.3],
                        [3, 1.1],
                        [3, 2.2],
                        [4, 2.2],
                        [4, 3.4],
                        [5, 0.3],
                        [5, 4.3],
                        [1, 1.2],
                        [1, 2.2],
                        [2, 3.5],
                        [2, 5.2],
                        [3, 3.1],
                        [3, 5.2],
                        [4, 5.5],
                        [4, 5.8],
                        [5, 5.8],
                        [5, 5.5]
                        ])
label_simple = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

x = data_simple
y = label_simple
knn = Knn(x, y)
# knn.plot_data(x, y)
# knn.plot_region(x, y, 7)
# knn.plot_region(x, y, 5)
# knn.plot_region(x, y, 3)
# knn.plot_region(x, y, 1)


# modelo com dados reais
c_data = datasets.load_wine()
x = c_data.data
y = c_data.target

# filtra apenas 2 tipos de vinho
x = x[0:130, :]
y = y[0:130]

# transformando y=0 em y=-1
y[y == 0] = -1

# dividir entre teste e treinamento
# train_size = round(len(x)*0.8)
# indexes = np.random.choice(x.shape[0], train_size, replace=False)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# x_train = x[indexes, :]
# y_train = y[indexes]
# x_test = x[round(len(x)*0.8):, :]
# y_test = y[not indexes]

# usar as de teste para avaliar a classificacão
knn_wd = Knn(x_train, y_train)
ks = range(1, 100)
accs = []
for k in ks:
    y_calculated = knn_wd.classify_many(x=x_test, k=k)
    correct = sum(y_calculated == y_test)
    acc = correct/len(y_test)
    print(acc)
    accs.append(acc)

fig = plt.figure()
plt.plot(accs)
# Add title and axis names
plt.title('Curva de acurácia')
plt.xlabel('K')
plt.ylabel('Acurácia')
plt.show()

# modelo com mais de 2 dimensões
c_data = datasets.load_breast_cancer()
x = c_data.data
y = c_data.target

# transformando y=0 em y=-1
y[y==0] = -1

# dividir entre teste e treinamento
x_train = x[0:round(len(x)*0.8), :]
y_train = y[0:round(len(y)*0.8)]
x_test = x[round(len(x)*0.8):, :]
y_test = y[round(len(y)*0.8):]

# usar as de teste para avaliar a classificacão
knn_bc = Knn(x_train, y_train)
ks = range(1, 100)
accs = []
for k in ks:
    y_calculated = knn_bc.classify_many(x=x_test, k=k)
    correct = sum(y_calculated == y_test)
    acc = correct/len(y_test)
    print(acc)
    accs.append(acc)

fig = plt.figure()
plt.plot(accs)
# Add title and axis names
plt.title('Curva de acurácia')
plt.xlabel('K')
plt.ylabel('Acurácia')
plt.show()

# iterar isso para alguns ks


# gerar a curva de eficácia do modelo


# transform data label into -1 and +1


# run for different k and show curves
