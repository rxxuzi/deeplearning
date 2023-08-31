import numpy as np

x1 = np.array([[1, 2], [3, 4]])
x2 = np.array([[5, 6], [7, 8]])
data = {'x1' : x1, 'x2' : x2}
np.savez('../step41~60/data.npz', **data)

arrays = np.load('../step41~60/data.npz')
x1 = arrays['x1']
x2 = arrays['x2']
print(x1)

from dezero import Layer
from dezero import  Parameter

layer = Layer()

l1 = Layer()
l1.p1 = Parameter(np.array(1))

layer.l1 = l1
layer.p2 = Parameter(np.array(2))
layer.p3 = Parameter(np.array(3))

params_dict = {}
layer._flatten_params(params_dict)
print(params_dict)