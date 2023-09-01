import contextlib
import weakref
import numpy as np
import dezero

class Config:
    enable_backprop = True #逆伝播を有効/無効
    train = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def test_mode():
    return using_config('train' , False)

def no_grad():
    return using_config('enable_backprop', False)


def test_mode():
    return using_config('train', False)

class Var:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.name = name  # 変数名
        self.data = data  # 数値
        self.grad = None  # 微分値
        self.creator = None  # 生成者
        self.generation = 0  # 世代

    # =====================================
    # + - * / ** をオーバーロード
    # =====================================

    def __len__(self):  # 要素数を取得
        return len(self.data)

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):  # 演算子のオーバーロード
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __truediv__(self, other):
        return div(self, other)
    def __pow__(self, power, modulo=None):
        return pow(self, power)
    def __neg__(self):
        return neg(self)

    def clean_grad(self):  # 微分値を消去
        self.grad = None

    def cleargrad(self):
        self.grad = None

    @property
    def shape(self):  # 多次元配列の形状を取得
        return self.data.shape

    @property
    def ndim(self):  # 次元の数を取得
        return self.data.ndim

    @property
    def size(self):  # 要素の数
        return self.data.size

    @property
    def dtype(self):  # データの型を取得
        return self.data.dtype

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 世代をインクリメント

    # BackWard
    def backward(self, retain_grad=False , create_graph = False):
        """
        :param retain_grad: 全ての変数が微分を保持するか(Falseの場合、微分値をリセット)
        :param create_graph: 逆伝播の処理をするか
        :return:
        """

        if self.grad is None:
            self.grad = Var(np.ones_like(self.data)) # 微分を自動補充

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakref

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self)

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)

    # toString
    def __repr__(self):
        if self.data is None:
            return 'Var(none)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'Var(' + p + ')'


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# objがVarインスタンスじゃない場合、Varインスタンスとして返す
def as_var(obj):
    if isinstance(obj, Var):
        return obj
    return Var(obj)


class Function:
    def __call__(self, *inputs):
        inputs = [as_var(x) for x in inputs]  # 各要素xをVarインスタンスに変換

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Var(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, xs):
        return -xs

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


# 割算
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


# 指数関数
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)

def setup():
    Var.__add__ = add
    Var.__radd__ = add
    Var.__mul__ = mul
    Var.__rmul__ = mul
    Var.__neg__ = neg
    Var.__sub__ = sub
    Var.__rsub__ = rsub
    Var.__truediv__ = div
    Var.__rtruediv__ = rdiv
    Var.__pow__ = pow
    Var.__getitem__ = dezero.functions.get_item

class Parameter(Var):
    pass