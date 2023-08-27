import numpy as np
from dezero import Var
from dezero import functions as F
import matplotlib.pyplot as plt


def plot_graph(x, y, title):
    plt.plot(x, y)
    plt.title(title)
    plt.show()
