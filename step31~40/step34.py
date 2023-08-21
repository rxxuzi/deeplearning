import numpy as np
from dezero import Var
import dezero.functions as F
from dezero.functions import sin, cos

x = Var(np.array(1.0))

y = F.sin(x)
print(y)

y.backward(create_graph=True)

for i in range(3):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    print(f"{i}階微分 : gx = {gx.data}, x = {x.data}")