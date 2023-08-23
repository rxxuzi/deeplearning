import numpy as np
from dezero import Var
import dezero.functions as F
from dezero.utils import sum_to

x = Var(np.array([[1, 2, 3], [4, 5, 6]]))

y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Var(np.random.randn(2,3,4,5))
y = x.sum(keepdims=True)
print(y.shape)