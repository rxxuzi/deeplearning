import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero import Model, Var

# DataSet
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# hyper parameters
lr = 0.2
max_iters = 10000
hidden_size = 10

#model
class MLP(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def  forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

x = Var(np.random.randn(5, 10),  name='x')
model = MLP(100, 10)
# model.plot(x)

# training
for i in range(max_iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)