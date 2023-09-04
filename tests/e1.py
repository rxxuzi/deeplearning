from dezero import Var , Function
import numpy as  np
from dezero.functions import mean_squared_error_simple

a = Var(np.array([1]))
b = Var(np.array([2]))

c = mean_squared_error_simple(a,b)
print(c)