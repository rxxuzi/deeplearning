import numpy as np
import pandas as pd

# class の実装
# Variableクラスは箱として使うことができる

class Variable :
    def __init__(self, data):
        self.data = data

# スカラ
data = np.array(1.9)
var = Variable(data)
print(var.data)

# ベクトル
data = np.array([1,2,3])
var = Variable(data)
print(var.data)

# 行列
data = np.array([[1,2,3],
                 [4,5,6]])
var = Variable(data)
print(var.data)


