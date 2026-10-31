import numpy as np

class MSE():
    def value(self, y_hat ,y):
        m = y.shape[0]
        return 1 / (2*m) * np.sum((y_hat - y) ** 2)

    def gradient(self, y_hat, y):
        m = y.shape[0]
        grad_pred = 1 / m  * (y_hat - y)
        return grad_pred

# .........
class RidgeLoss():
    def value(self, X, w, b ,y, alpha):
        m = X.shape[0]
        return 1 / (m*2) * np.sum((X @ w + b - y)**2) + alpha / 2 * w.T @ w
    def gradient(self, X, w, b, y, alpha):
        m = X.shape[0]
        y_hat = X @ w + b
        dw = 1 / m * X.T @ (y_hat- y) + alpha * w
        db = 1 / m * np.sum(y_hat - y) 
        return dw, db

class LassoLoss():
    def value(self, X, w, b ,y, alpha):
        m = X.shape[0]
        return 1 / (2*m) * np.sum((X @ w + b - y)**2) + alpha * np.linalg.norm(w, ord = 1)
    def gradient(self, X, w, b, y, alpha):
        m = X.shape[0]
        y_hat = X @ w + b 
        dw = 1 / m * X.T @ (y_hat- y) + alpha * np.sign(w)
        db = 1 / m * np.sum(y_hat - y) 
        return dw, db

class ElasticLoss():
    def value(self, X, w, b, y, alpha, beta):
        m = X.shape[0]
        return 1 / (2*m) * np.sum((X@w + b - y)**2) + alpha/2 * np.linalg.norm(w, ord = 2)**2 + beta * np.linalg.norm(w, ord = 1)
    def gradient(self, X, w, b, y, alpha, beta):
        m = X.shape[0]
        y_hat = X @ w + b
        dw = 1 / m * X.T @ (y_hat - y) + alpha * w + beta * np.sign(w)
        db = 1 / m * np.sum(y_hat - y)
        return dw, db