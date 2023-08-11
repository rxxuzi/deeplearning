#　自動的に微分する

import  numpy as np

class Var :
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator #関数を取得
        if f is not None:
            x = f.input # 関数の入力を取得
            x.grad = f.backward(self.grad) # 関数のbackwardメソッドを呼ぶ
            x.backward() #　関数のbackwardメソッドを実行(再帰)

class Function :
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Var(y)
        output.set_creator(self) #出力変数に生みの親を覚えさせる
        self.input = input
        self.output = output #出力も覚える
        return output

    def forward(self, x):
        raise NotImplementedError()
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


A  = Square()
B  = Exp()
C  = Square()
x  = Var(np.array(0.5))
a  = A(x)
b  = B(a)
y  = C(b)

#逆向きに計算ブラフのノードを辿る
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x


y.grad = np.array(1.0)
y.backward()
print(x.grad)


