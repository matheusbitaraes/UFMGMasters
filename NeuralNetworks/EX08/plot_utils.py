import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')


def plot_result(data, labels, title="Test"):
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.boxplot(data)
    plt.show()
    return None