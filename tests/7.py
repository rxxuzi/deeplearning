from dezero import Plot

import numpy as np
logs = []

def f(x):
    return np.sin(x)


length = 1000
start = - length/2
weight = 0.01
d = 1/length
for i in range(length):
    logs.append(np.sin(i*weight))
    print(logs[i])


p = Plot(len(logs), logs)
p.display()