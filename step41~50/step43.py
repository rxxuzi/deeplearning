import numpy as np
from dezero import Var
from dezero import functions as F
from dezero import utils


# 線形変換
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)  # rand is noise

# # 線形回帰の結果をグラフにプロット
import matplotlib.pyplot as plt
plt.scatter(x, y, s = 10,label="Actual Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Non-linear Data Sets")
plt.legend()
plt.show()