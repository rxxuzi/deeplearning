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
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

#関数を用意
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
# 合成関数
a = A(x)
b = B(a)
y = C(b)
print(y.data)