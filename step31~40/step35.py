# tanh
import numpy as np
from dezero import Var
import dezero.functions as F
from  dezero.utils import plot_dot_graph

x = Var(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 2

for i in range(iters):
    gx = x.grad
    print(gx)
    x.cleargrad()
    gx.backward(create_graph = True)

gx = x.grad
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')