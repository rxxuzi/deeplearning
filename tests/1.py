from dezero import Var
import numpy as np
from dezero import *

x = Var(np.array(4.0))
y = x ** 2 + 1
print(y.data)
y.backward(create_graph=True)
print(x.grad)
gx = x.grad
x.cleargrad()
gx.backward()
print(x.grad)