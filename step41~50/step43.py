import numpy as np

from dezero import Var
from dezero import functions as func

# データセット
np.random.seed(0)
x = np.random.rand(200, 1)
# sin(2πx) + noise
y = np.sin(2 * np.pi * x) + np.random.rand(200, 1)

# 1. 重みの初期化
I = 1   # the number of input units
H = 10  # the number of hidden units
O = 1   # the number of output unit
small = 0.01
# weight initialization
W1 = Var(small * np.random.randn(I, H))
W2 = Var(small * np.random.randn(H, O))
# bias initialization
b1 = Var(np.zeros(H))
b2 = Var(np.zeros(O))

# 2. ニューラルネットワークの推論
def predict(x):
    y = func.linear(x, W1, b1) # linear transformation
    y = func.sigmoid(y)        # activation function
    y = func.linear(y, W2, b2) # linear transformation
    return y

lr = 0.2       # learning rate
iters = 10000  # the number of iteration

# 3. ニューラルネットワークの学習
for i in range(iters):
    y_pred = predict(x)
    loss = func.mean_squared_error(y, y_pred)
    # 微分値の初期化
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % (iters/10) == 0:
        print(W1, b1, W2, b2, loss)

# プロット
import matplotlib.pyplot as plt

plt.title("Non-linear Data Sets")

plt.scatter(x, y, s=10, label="Actual Data")
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r', label="Predicted Line")
plt.legend()
plt.show()
