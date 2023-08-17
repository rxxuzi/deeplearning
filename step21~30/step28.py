# ローゼンブロック関数
# Code snippet from step27.py

from dezero import Var
import numpy as np

def  rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = Var(np.array(0.0))
x1 = Var(np.array(2.0))

lr = 0.001 # 学習率
iter = 1000 # 試行回数


for i in range(iter):
    print(x0.data , x1.data)
    y = rosenbrock(x0, x1)

    x0.clean_grad()
    x1.clean_grad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

    if i % 100 == 0:
        print(x0, x1)