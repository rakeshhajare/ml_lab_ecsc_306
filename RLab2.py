import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('/home/jsr/v.txt')
X_data = data[:,:2]
Y_data = data[:,2]
print(data)
print(X_data)
print(Y_data)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def predict(x, W, b):
    logits = np.matmul(x, W) + b
    return 1 * (logits >= 0)

def compute_accuracy(x, W, b, y):
    labels = predict(x, W, b)
    return np.mean(labels == y)

def compute_gradients(x, W, b, y):
    logits = np.matmul(x, W) + b
    y_pred = sigmoid(logits)
    error  = y_pred - y

    dLdW  = np.mean(error * x.T, axis=1)
    dLdb  = np.mean(error)

    return dLdW, dLdb


learning_rate = 0.005
num_epochs    = 100


W = np.zeros(2)
b = 0.0

np.random.seed(0)


for epoch in range(num_epochs):
 
    idx = np.random.permutation(data.shape[0])
    for i in idx:
        grads = compute_gradients(X_data[i:i+1], W, b, Y_data[i:i+1])
        W -= learning_rate * grads[0]
        b -= learning_rate * grads[1]


    if (epoch+1) % 10 == 0:
        accuracy = compute_accuracy(X_data, W, b, Y_data)
        print("After {} epochs, accuracy = {}".format(epoch+1, accuracy))


print("W =", W)
print("b =", b)



labels = predict(X_data, W, b)


idx_0, = np.where(labels == 0)
idx_1, = np.where(labels == 1)


plt.plot(X_data[idx_0,0], X_data[idx_0,1], 'bo', label='I. versicolor')
plt.plot(X_data[idx_1,0], X_data[idx_1,1], 'ro', label='I. virginica')

x_sep = np.linspace(X_data[:,0].min(), X_data[:,0].max())
y_sep = (-b - W[0]*x_sep) / W[1]
plt.plot(x_sep, y_sep)


plt.legend()

plt.show()

   

