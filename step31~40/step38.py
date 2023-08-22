import numpy as np
from dezero import Var
import dezero.functions as F

x = Var(np.random.rand(2,3))
y = x.transpose()
y = x.T
print(y)