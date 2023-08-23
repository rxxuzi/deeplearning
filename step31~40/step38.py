import numpy as np
from dezero import Var
import dezero.functions as F

x = Var(np.array([[1,2,3],[4,5,6]]))
y = x.transpose()
y = x.T
print(y)

A,B,C,D = 1,2,3,4
x = np.random.rand(A,B,C,D)
print(x)
y = x.transpose(1,0,3,2)
print(y)