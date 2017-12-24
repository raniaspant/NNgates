import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def sigmoid(a):  # defining the sigmoid function
    return 1 / (1 + np.exp(-a))


def y_function(x1, x2, x3, w1, w2, w3):  # y = f(a) = 1/(1 + e^(-a))
    a = w1 * x1 + w2 * x2 + w3 * x3  # defining a
    return sigmoid(a)


X = np.matrix([[0.1, 0.1, 1], [0.1, 0.9, 1], [0.9, 0.1, 1], [0.1, 0.9, 1]])
# X.item(i,j), i = 0:4, j = 0:3
# D = np.matrix([[0.1], [0.9], [0.9], [0.9]])  # desired outputs for OR gate
D = np.matrix([[0.1], [0.9], [0.9], [0.9]])    # desired outputs for AND gate
w1 = np.random.randn()
w2 = np.random.randn()
w3 = np.random.randn()
Error = np.matrix([[10.0], [10.0], [10.0], [10.0]])
for i in range(0, 4):
    y = y_function(X.item(i, 0), X.item(i, 1), X.item(i, 2), w1, w2, w3)
    Error[i] = np.power((y - D.item(i)), 2)

TotalError = np.sum(Error)
# zero 1 list for each dw
dw1 = np.zeros((4, 1))
dw2 = np.zeros((4, 1))
dw3 = np.zeros((4, 1))
e = []
w1p = []
w2p = []
w3p = []
e.append(TotalError)
w1p.append(w1)
w2p.append(w2)
w3p.append(w3)

while TotalError > 0.001:
    w1 = w1 + np.sum(dw1)
    w2 = w2 + np.sum(dw2)
    w3 = w3 + np.sum(dw3)
    for i in range(0, 4):
        y = y_function(X.item(i, 0), X.item(i, 1), X.item(i, 2), w1, w2, w3)
        Error[i] = np.power((y - D.item(i)), 2)
        dw1[i] = 2 * (D.item(i) - y) * y * (1 - y) * X.item(i, 0)
        dw2[i] = 2 * (D.item(i) - y) * y * (1 - y) * X.item(i, 1)
        dw3[i] = 2 * (D.item(i) - y) * y * (1 - y) * X.item(i, 2)
    TotalError = np.sum(Error)
    e.append(TotalError)
    w1p.append(w1)
    w2p.append(w2)
    w3p.append(w3)

# Next lines are all about the plots required
plt.plot(e, '*r-')                              # error plot
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.show()
w1_plot, = plt.plot(w1p, '*r-', label='W1')   # weights plot
w2_plot, = plt.plot(w2p, 'xb-', label='W2')
w3_plot, = plt.plot(w3p, 'og-', label='W3')
plt.legend(handler_map={w1_plot: HandlerLine2D(numpoints=4)})
plt.ylabel('Weight')
plt.xlabel('Iterations')
plt.show()
