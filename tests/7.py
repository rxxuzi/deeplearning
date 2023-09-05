import  numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def htan(x):
    return np.tanh(x)

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x1 = np.arange(-2.5, 2.5, 0.05)
y = []
label = ['relu',  'sigmoid', 'tanh' , 'leaky relu']
color = ['red', 'blue', 'green', 'yellow']

y1 = relu(x1)
y2 = sigmoid(x1)
y3 = htan(x1)
y4 = leaky_relu(x1)

line1 = ax.plot(x1,y1,label=label[0] , color=color[0]) #axグラフを描く
line2 = ax.plot(x1,y2,label=label[1] , color=color[1])
line3 = ax.plot(x1,y3,label=label[2] , color=color[2])
line4 = ax.plot(x1,y4,label=label[3] , color=color[3])

ax.legend(loc='upper left')
ax.spines["right"].set_color("none")      # グラフ右側の軸(枠)を消す
ax.spines["top"].set_color("none")        # グラフ上側の軸(枠)を消す
ax.spines['left'].set_position('zero')    #グラフ左側の軸(枠)が原点を通る
ax.spines['bottom'].set_position('zero')  #グラフ下側の軸(枠)が原点を通る
plt.grid()
plt.show()