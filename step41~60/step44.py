# from dezero.layers import *
from dezero.core import *
import numpy as np

import dezero.layers as L

import dezero.functions as F

# データセット
np.random.seed(0)
x = np.random.rand(200, 1)
# sin(2πx) + noise
y = np.sin(2 * np.pi * x) + np.random.rand(200, 1)

# 1. 出力サイズを指定
l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)

# プロット
import matplotlib.pyplot as plt

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
