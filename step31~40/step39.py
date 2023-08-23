import numpy as np
from dezero import Var
import dezero.functions as F

x = Var(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x, )