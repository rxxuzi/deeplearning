if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Var

def f(a):
    return a ** 4 - 2 * a ** 2

x = Var(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    y  = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad
    x.data -= gx.data / gx2.data
    # print(x.grad)