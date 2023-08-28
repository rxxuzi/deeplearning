# MINIST
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero.datasets import Spiral
from dezero import  DataLoader
from dezero import Progress

max_epoch = 5
batch_size = 100
hidden_size = 100

def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x


train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10),  activation=F.relu)
# optimizer = optimizers.SGD().setup(model)
optimizer = optimizers.MomentumSGD().setup(model)

labels = ["train","test"]
trls = [] # Train Loss
tels = [] # Test Loss

def ave(sum):
    return float(sum / len(train_set))

# Training
for epoch in range(max_epoch):
    sum_loss = 0
    sum_acc  = 0

    # train
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    trls.append(ave(sum_loss))

    print('epoch %d \n train : loss %.4f, accuracy %.4f' % (epoch + 1, sum_loss / len(train_set), sum_acc / len(train_set)))


    sum_loss = 0
    sum_acc = 0
    with dezero.no_grad():
        #
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        tels.append(ave(sum_loss))


    print(' test  : loss %.4f, accuracy %.4f' % (sum_loss / len(test_set), sum_acc / len(test_set)))

logs = [trls , tels]

# =============================================================================
# plot data
# =============================================================================
import matplotlib.pyplot as plt

for i,  label in enumerate(logs):
    plt.plot(range(1, max_epoch + 1), logs[i], label=labels[i])
# plt.plot(range(1, max_epoch + 1), logs)
plt.title("Train Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc = 'lower right')
plt.show()

# =============================================================================
print("finish")