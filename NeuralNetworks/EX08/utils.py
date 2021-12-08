import numpy as np
from sklearn.model_selection import train_test_split


def multiple_executions(X, Y, model, num_executions=30):
    X = normalize(X)
    acc_list = np.zeros([num_executions, 1])
    n_rows = X.shape[0]
    n_columns = X.shape[1] + 1
    Y.reshape([n_rows, 1])
    dataset = np.c_[X, Y]
    for i in range(0, num_executions):
        training_data, testing_data = train_test_split(dataset, test_size=0.3)
        model.train(training_data)
        acc_list[i, 0] = model.get_accuracy_and_error(testing_data)
    return acc_list


def normalize(dataset):
    dataset_normalized = (dataset - dataset.min(axis=0)) / (dataset.max(axis=0) - dataset.min(axis=0))

    return dataset_normalized


def print_mean_std(data, model, dataset_name):
    name = model.get_name()
    mean = np.mean(data)
    std = np.std(data)

    print(f"\n --- dataset: {dataset_name} -- model {name}, mean: {mean}, std:{std} --- \n")
    return None


