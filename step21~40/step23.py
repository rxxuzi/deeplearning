# 偏微分

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Var

def sphere (x , y):
    z = x ** 2 + y ** 2
    return z

def matyas (x , y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

x = Var(np.array(1.0))
y = Var(np.array(1.0))
z = matyas(x, y)
z.backward()
print(z.data)
print(x.grad , y.grad)