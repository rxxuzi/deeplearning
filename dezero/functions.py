import numpy as np
from dezero.core import *
from dezero import utils

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
    def forward(self, x):
        y = np.transpose(x)
        return y

    def backward(self, gys):
        return transpose(gys)

def transpose(x):
    return Transpose()(x)

# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================

class Sum(Function):
    def __init__(self, axis , keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)

def sum(x, axis = None , keepdims = None):
    return Sum(axis,keepdims)(x)

class BroadcastTo(Function):
    def __init__(self,shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x,self.shape)

    def backward(self, gy):
        return sum_to(gy, self.x_shape)

def broadcast_to(x,shape):
    if x.shape == shape:
        return as_var(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)

    def backward(self, gy):
        return broadcast_to(gy,self.x_shape)


def sum_to(x,shape):
    if x.shape == shape:
        return as_var(x)
    return SumTo(shape)(x)