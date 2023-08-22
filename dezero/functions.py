import numpy as np
from dezero.core import *


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================

class Sin(Function):
    def forward(self, xs):
        y = np.sin(xs)
        return y

    def backward(self, gys):
        x, = self.inputs
        gx = gys * cos(x)
        return  gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, xs):
        y = np.cos(xs)
        return y

    def backward(self, gys):
        x, = self.inputs
        gx = gys * sin(x) * -1
        return gx

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, xs):
        y = np.tanh(xs)
        return y

    def backward(self, gys):
        y = self.outputs[0]()
        gx = gys * (1 - y * y)
        return gx

def tanh(x):
    return Tanh()(x)

# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    """
    テンソルを整形するクラス
    """
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_var(x)
    return Reshape(shape)(x)

class Transpose(Function):
    """
    行列を転置するクラス
    """
    def forward(self, x):
        return np.transpose(x)

    def backward(self, gy):
        return transpose(gy)

def transpose(x):
    return Transpose()(x)