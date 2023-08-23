from dezero import Var
import dezero.functions as F
import numpy as np
from dezero.utils import sum_to

x0 = Var(np.array([1,2,3]))
x1 = Var(np.array([10]))
y = x0 + x1
print(y)

y.backward()
print(x1.grad)