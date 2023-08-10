import numpy as np


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

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y


x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(y))
print(y.data)
