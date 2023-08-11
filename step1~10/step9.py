# より実用的に
import numpy as np


class Var:
    def __init__(self, data):

        # if data is not None:
        #     if not isinstance(data, np.ndarray):
        #         raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # gradがnoneの場合自動的に微分を生成

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 関数を取得
            x, y = f.input, f.output  # 関数の入力の要素を取得
            x.grad = f.backward(y.grad)  # 関数のbackwardメソッドを呼ぶ
            if x.creator is not None:
                funcs.append(x.creator)  # 1つ前の関数をリストに追加

#スカラをベクトル(配列)にするメソッド
def as_array(self, y):
    if np.isscalar(y):
        return np.array(y)
    return y


class Function:

    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Var(as_array(y))
        output.set_creator(self)  # 出力変数に生みの親を覚えさせる
        self.input = input
        self.output = output  # 出力も覚える
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


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


x = Var(np.array(0.5))
y = square(exp(square(x)))
print(f"y is {y.data}")
# y.grad = np.array(1.0)
y.backward()
print(f"x.grad is {x.grad}")
