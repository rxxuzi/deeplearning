import numpy as np
from dezero.core import Function

class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x, = self.inputs
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        x, = self.inputs
        gx = -gy * np.sin(x)
        return gx


def cos(x):
    return Cos()(x)

