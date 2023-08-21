import numpy as np
from dezero import Var
import dezero.functions as F

x = Var(np.array(1.0))
y = F.sin(x)
y.backward(create_graph=True)

print(x.grad)