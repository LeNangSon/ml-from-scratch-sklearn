import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class BCE():
    def value(self, X, w, b, y):
        logit = X @ w + b
        y_hat = sigmoid(logit)
        return -np.mean((1-y) * np.log(1 - y_hat) + y * np.log(y_hat))
    def gradient(self, X, w, b, y):
        m = X.shape[0]
        logit = X @ w + b
        y_hat = sigmoid(logit)
        grad_w = 1/ m * (X.T @ (y_hat - y))
        grad_b = np.mean(y_hat - y)
        return grad_w, grad_b