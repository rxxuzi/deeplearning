# 行列の積
# ベクトルの内積
import numpy as np
#ベクトルの内積
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.dot(a,b)
print(c)

#行列の積
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c = np.dot(a,b)
print(c)

#行列の積では対応する次元の要素数を一致させる必要がある

from dezero import Var
import dezero.functions as F

x = Var(np.random.randn(2,3))
W = Var(np.random.randn(3,4))
y = F.matmul(x,W)
y.backward()

print(x.grad.shape)
print(W.grad.shape)

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

