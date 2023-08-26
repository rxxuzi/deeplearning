import dezero
import numpy as np
x, t = dezero.datasets.get_spiral()
print(x.shape, t.shape)

print(x.max(), x.min())
print(t.max(), t.min())

import matplotlib.pyplot as plt
plt.scatter(x[:,0], x[:,1], c=t, s=10)
plt.show()

max_epoch = 100
batch_size = 30
hidden_size = 10
lr = 1.0

model = dezero.models.MLP((hidden_size, 3))
optimizer = dezero.optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = data_size // batch_size

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0
    for i in range(max_iter):
        batch_index = index[i*batch_size:(i+1)*batch_size]
        batch = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch)
        loss = dezero.functions.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    # print('epoch %d, loss %f' % (epoch+1, avg_loss))

# ==========================================================
# Plot boundary area the model predict
# ==========================================================

# Set hyperparameters
h = 0.001
# Set boundary area of the model predict
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

# Predict class of the model
with dezero.no_grad():
    score = model(X)

# Convert to one-hot
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

# Plot data points of the dataset
# N, CLS_NUM = 100,3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40,  marker=markers[c], c=colors[c])
plt.show()