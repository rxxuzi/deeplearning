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
# import torch.nn.functional as F

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

progress = Progress(max_epoch, bar_width=30)
train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

train_loss_logs = []
train_acc_logs = []
test_loss_logs = []
test_acc_logs = []

loss_labels = ['train loss', 'test loss']
acc_labels = ['train acc', 'test acc']

for epoch in range(max_epoch):
    sum_loss , sum_acc = 0, 0

    for x, t in train_loader:
        y  = model(x) # 1. 訓練用のミニバッチデータ
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t) # 2. 訓練データの認識制度
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

        avg_loss = sum_loss / len(train_set)
        avg_acc = sum_acc / len(train_set)

        train_loss_logs.append(avg_loss)
        train_acc_logs.append(avg_acc)

    # Print loss every epoch
    print('epoch %d, loss %.2f, acc %.2f' % (epoch + 1, sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss , sum_acc = 0, 0
    with dezero.no_grad(): #　3. 勾配不要モード
        for x,t in test_loader: # 4. 訓練用のミニバッチデータ
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)  # 2. 訓練データの認識制度
            sum_loss += float(loss.data) * len(t)
            sum_acc  += float(acc.data)  * len(t)

            avg_loss = sum_loss / len(test_set)
            test_loss_logs.append(avg_loss)
            avg_acc = sum_acc / len(test_set)
            test_acc_logs.append(avg_acc)
    # Print loss every epoch
    print('test loss %.2f, acc %.2f' % (sum_loss / len(test_set), sum_acc / len(test_set)))

    progress.update_bar(epoch+1)

plt.plot(range(1, max_epoch + 1), train_loss_logs, label=loss_labels[0])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc='lower right')
plt.show()