import pandas as pd
from Perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

# leitura dos dados
df = pd.read_csv('data/breast-cancer-wisconsin.data', header=0, names=["id", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
                                                                     "f8", "f9", "class"])

# limpeza dos dados, removendo colunas que não possuírem os dados completos
df = df.dropna(thresh=1)
df = df.drop(df.loc[df['f6'] == '?'].index)
df = df.astype(int)

# plot dos primeiros dados
df.head()

# Realizando a troca de valores do resultado
df.loc[df['class'] == 2, 'class'] = 1
df.loc[df['class'] == 4, 'class'] = 0
print(df.head())

accuracy_train_list = []
accuracy_test_list = []
for i in range(0, 20):
    # dados de treinamento
    df_train = df.sample(frac=0.7)  # random state is a seed value
    # df_train['id'].count()

    # dados de teste
    df_test = df.drop(df_train.index)
    x_test = df_test.iloc[1:, 1:10].values
    y_test = df_test.iloc[1:, 10].values
    # df_test['id'].count()

    ppn = Perceptron(epochs=100, learning_rate=0.5, verbose=False)
    x_train = df_train.iloc[1:, 1:10].values
    y_train = df_train.iloc[1:, 10].values

    # Apenas para substituir as classes 0 e 1 para 1 e -1
    # A classe 1 é a que desejamos encontrar.

    trained_ppn = ppn.fit(x_train, y_train)
    y_train_result = trained_ppn.predict(x_train)

    # a operação que compara os resultados é 1 - (yteste-yt/num_amostras_teste)
    accuracy_train = 1 - np.dot((y_train-y_train_result).transpose(), (y_train-y_train_result))/len(y_train)
    accuracy_train_list.append(accuracy_train)

    y_test_result = trained_ppn.predict(x_test)
    accuracy_test = 1 - np.dot((y_test-y_test_result).transpose(), (y_test-y_test_result))/len(y_train)
    accuracy_test_list.append(accuracy_test)

plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(x=accuracy_train_list, bins=10, color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Acurácia das amostras de treinamento',fontsize=15)
plt.show()

plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(x=accuracy_test_list, bins=10, color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Acurácia das amostras de teste',fontsize=15)
plt.show()

print(np.mean(accuracy_train_list))
print(np.std(accuracy_train_list))
print(np.mean(accuracy_test_list))
print(np.std(accuracy_test_list))
