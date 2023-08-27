import numpy as np
from dezero import Var
from dezero import functions as F
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, x, y, lb_x = "x",  lb_y = "y", title = "Graph"):
        self.x = x
        self.y = y
        self.lb_x = lb_x
        self.lb_y = lb_y
        self.title = title
        self.labels = None

    def label(self, lb_x, lb_y, title):
        self.lb_x = lb_x
        self.lb_y = lb_y
        self.title = title
    def set_label(self, lb_x, lb_y, title):
        self.lb_x = lb_x
        self.lb_y = lb_y
        self.title = title

    def set_data(self, x, y):
        self.x = x
        self.y = y

    def display(self):
        if self.labels == None:
            plt.plot(self.x, self.y)
        else:
            for i in enumerate(y):
                plt.plot(self.x, self.y[i], label = self.labels[i])

        plt.xlabel(self.lb_x)
        plt.ylabel(self.lb_y)
        plt.title(self.title)
        plt.show()

def plot_graph(x,y):
    p = Plot(x,y)


class Progress:
    def __init__(self, max, progress = 0,percentage = 10 , bar_width = 20):
        self.max = max
        self.percentage = percentage
        self.progress = progress
        self.total_width = bar_width

    def update(self, epoch):
        self.progress = (epoch) / self.max
        self.percentage = self.progress * 100

    def progress_bar(self):
        if self.progress < 0 or self.progress > 1:
            print("p should be between 0 and 1")
            return

        filled_width = int(self.progress * self.total_width)
        empty_width = self.total_width - filled_width

        if self.progress == 1:
            bar = "[" + "#" * filled_width + "=" * empty_width + "]" + "100.00%"
        else:
            bar = "[" + "#" * filled_width + "=" * empty_width + "]" + f"{self.percentage:.2f}%"

        print(bar)

    def update_bar(self, epoch):
        """

        :param epoch: 進捗数
        :return: progress_bar()
        """
        self.progress = (epoch) / self.max
        self.percentage = self.progress * 100
        self.progress_bar()
