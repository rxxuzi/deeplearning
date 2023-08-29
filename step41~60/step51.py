# MINIST
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero
import dezero.functions as F
from dezero import DataLoader
from dezero import optimizers
from dezero.models import MLP

# Hyperparameters
max_epoch = 50
batch_size = 100
hidden_size = 200

def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x

# Datasets
train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# Model
# Activation function is relu
model = MLP((hidden_size, 10),  activation=F.relu)
# optimizer = optimizers.SGD().setup(model)
# Optimizer
optimizer = optimizers.MomentumSGD().setup(model)

labels = ["train","test"]
trls = [] # Train Loss
tels = [] # Test Loss

# Training
for epoch in range(max_epoch):

    sum_loss = 0
    sum_acc  = 0

    # train
    for x, t in train_loader:
        # Forward Propagation
        y = model(x)
        # Calculate loss and accuracy
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        # Backpropagation
        model.cleargrads()
        loss.backward()
        # Update parameters
        optimizer.update()
        # Accumulate loss and accuracy
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    ave_loss = sum_loss / len(train_set)
    trls.append(ave_loss)

    print('epoch : %d \n train : loss %.4f, accuracy %.4f' % (epoch + 1, sum_loss / len(train_set), sum_acc / len(train_set)))

    # Reset
    sum_loss = 0
    sum_acc = 0
    # test
    with dezero.no_grad():
        for x, t in test_loader:
            # Forward Propagation
            y = model(x)
            # Calculate loss and accuracy
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            # Accumulate loss and accuracy
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        ave_loss = sum_loss / len(test_set)

        tels.append(ave_loss)


    print(' test  : loss %.4f, accuracy %.4f' % (sum_loss / len(test_set), sum_acc / len(test_set)))

logs = [trls , tels]

# =============================================================================
# plot data
# =============================================================================
import matplotlib.pyplot as plt

for i,  label in enumerate(logs):
    plt.plot(range(1, max_epoch + 1), logs[i], label=labels[i])

plt.title("Train Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc = "upper right")
plt.show()

# =============================================================================
print("finish")
