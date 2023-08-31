if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dezero
import numpy as np
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

dropout_ratio = 0.6
x = np.ones(10)

mask = np.random.rand(10) > dropout_ratio
y = x * mask
print(y)

scale = 1 - dropout_ratio
y = x * scale
print(y)
