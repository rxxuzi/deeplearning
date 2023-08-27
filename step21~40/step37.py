import numpy as np
import dezero.functions as F
from dezero import Var

x = Var(np.array([[1,2,3],[4,5,6]]))
c = Var(np.array([[10,20,30],[40,50,60]]))
y = F.sin(x)
t = x + c
print(y)
y = F.sum(t)