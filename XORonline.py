import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def sigmoid(a):  # defining the sigmoid function
    return 1 / (1 + np.exp(-a))


def y_function(x1, x2, x3, w1, w2, w3):  # y = f(a) = 1/(1 + e^(-a))
    a = w1 * x1 + w2 * x2 + w3 * x3  # defining a
    return sigmoid(a)


def column(matrix, i):
    return np.asarray([row[i] for row in matrix])


# inputs for XOR gate
X = np.array([[0.1, 0.1, 1], [0.1, 0.9, 1], [0.9, 0.1, 1], [0.9, 0.9, 1]])
# desired outputs for XOR gate
D = np.array([[0.1], [0.9], [0.9], [0.1]])
E = np.array([[10.0], [10.0], [10.0], [10.0]])
# weights for the first layer : 6 in total
W = np.random.random((3, 2))
# weights for the second layer
Y = np.random.random((4, 1))
Wbar = np.random.random((3, 1))

DW = np.ones((4, 6))
DWbar = np.ones((4, 3))
average = np.mean(E)
e = []
w1l2 = []
w2l2 = []
w3l2 = []
w1l1 = []
w2l1 = []
w3l1 = []
w4l1 = []
w5l1 = []
w6l1 = []
w1l2.append(Wbar[0][0])
w2l2.append(Wbar[1][0])
w3l2.append(Wbar[2][0])
w1l1.append(W[0][0])
w2l1.append(W[0][1])
w3l1.append(W[1][0])
w4l1.append(W[1][1])
w5l1.append(W[2][0])
w6l1.append(W[2][1])
while average > 0.001:
    for n in range(0, 4):
        while E[n] > 0.001:
            layer1 = sigmoid(np.dot(X, W))
            Z = np.ones((4, 3))
            Z[:, 0] = np.array([column(layer1, 0)])
            Z[:, 1] = np.array([column(layer1, 1)])
            # third column is ones and acts as threshold value 1 for z3
            layer2 = sigmoid(np.dot(Z, Wbar))
            Y = layer2
            E = np.power(Y - D, 2)
            for j in range(0, 3):
                DWbar[n][j] = -2 * (Y[n] - D[n]) * Y[n] * (1 - Y[n]) * Z[n][j]
                Wbar[j] += DWbar[n][j]
            k = 0
            for j in range(0, 3):
                for i in range(0, 2):
                    DW[n][k] = -2 * X[n][j] * Z[n][i] * (1 - Z[n][i]) * Wbar[i] * (Y[n] - D[n]) * Y[n] * (1 - Y[n])
                    k += 1
            k = 0
            for i in range(0, 3):
                for j in range(0, 2):
                    W[i][j] += DW[n][k]
                    k += 1
            layer1 = sigmoid(np.dot(X, W))
            Z[:, 0] = np.array([column(layer1, 0)])
            Z[:, 1] = np.array([column(layer1, 1)])
            # third column is ones and acts as threshold value 1 for z3
            layer2 = sigmoid(np.dot(Z, Wbar))
            Y = layer2
            E = np.power(Y - D, 2)
            w1l2.append(Wbar[0][0])
            w2l2.append(Wbar[1][0])
            w3l2.append(Wbar[2][0])
            w1l1.append(W[0][0])
            w2l1.append(W[0][1])
            w3l1.append(W[1][0])
            w4l1.append(W[1][1])
            w5l1.append(W[2][0])
            w6l1.append(W[2][1])
        layer1 = sigmoid(np.dot(X, W))
        Z = np.ones((4, 3))
        Z[:, 0] = np.array([column(layer1, 0)])
        Z[:, 1] = np.array([column(layer1, 1)])
        # third column is ones and acts as threshold value 1 for z3
        layer2 = sigmoid(np.dot(Z, Wbar))
        Y = layer2
        E = np.power(Y - D, 2)
        average = np.mean(E)
        e.append(average)
# Next lines are all about the plots required
plt.figure(1)
plt.subplot(311)
plt.plot(e, 'r-')                              # error plot
plt.ylabel('Error')
plt.xlabel('Iterations')
# plt.show()
plt.subplot(312)
w1_plot, = plt.plot(w1l2, '*r-', label='W1')   # weights plot
w2_plot, = plt.plot(w2l2, 'xb-', label='W2')
w3_plot, = plt.plot(w3l2, 'og-', label='W3')
plt.legend(handler_map={w1_plot: HandlerLine2D(numpoints=4)})
plt.ylabel('Weights of second layer')
plt.xlabel('Iterations')
plt.subplot(313)
w11_plot, = plt.plot(w1l1, '*-', label='W1')   # weights plot
w21_plot, = plt.plot(w2l1, 'x-', label='W2')
w31_plot, = plt.plot(w3l1, 'o-', label='W3')
w41_plot, = plt.plot(w4l1, 'd-', label='W4')   # weights plot
w51_plot, = plt.plot(w5l1, '_-', label='W5')
w61_plot, = plt.plot(w6l1, '|-', label='W6')
plt.legend(handler_map={w1_plot: HandlerLine2D(numpoints=4)})
plt.ylabel('Weights of first later')
plt.xlabel('Iterations')
plt.show()
