# プロット
import matplotlib.pyplot as plt
import numpy as np

title = "Linear Regression"
plt.title(title)

plt.scatter(x, y, s=10, label="Actual Data")
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r', label="Predicted Line")
plt.legend()
plt.show()

class Graph():

    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)