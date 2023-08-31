if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dezero
import numpy as np
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP
from dezero import test_mode

dropout_ratio = 0.6
x = np.ones(10)
print(x)

#学習時
y = F.dropout(x)
print(y)

#テスト時
with test_mode():
    y = F.dropout(x)
    print(y)
