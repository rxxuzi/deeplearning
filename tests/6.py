import math

import numpy as np

import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

max = 10000
w = np.pi * 6
x = np.array([ w * i/max for i in range(max)])
y = f(x)
plt.plot(x, y)
plt.show()