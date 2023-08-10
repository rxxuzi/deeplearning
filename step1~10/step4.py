import numpy as np

# 数値微分クラス


class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) #計算はforwardメソッドで行う
        output = Variable(y) #計算結果をVariableクラスに詰める
        return output

    def forward(self, x):
        raise NotImplementedError() #オーバーライドして使うメソッド

# Functionクラスをオーバーライドしたもの
class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

"""

"""
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

#関数を用意
# f(x) = x^2
# lim[h->0](f(x) + f(x+h)) / 2h
# を計算
f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)

#合成関数
# f(x) = (e^(x^2))^2
# A(x) = x
# B(x) = e^(x)
# C(x) = x^2
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)