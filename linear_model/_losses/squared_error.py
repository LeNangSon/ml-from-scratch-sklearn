import numpy as np

class MSE():
    def value(self, X, w, b ,y):
        m = X.shape[0]
        return 1 / m * np.sum((X @ w - y + b) ** 2)

    def gradient(self, X, w, b, y):
        m = X.shape[0]
        y_hat = X @ w + b
        grad_w = 2 / m * X.T @ (y_hat - y)
        grad_b = 2 / m * np.sum(y_hat - y)
        return grad_w, grad_b