from typing import Literal
from _solvers import SGD, Batch_GD, Mini_Batch_GD
from _losses import BCE
import numpy as np
import matplotlib.pyplot as plt

Solvers = Literal[
            "batch_gd",
            "mini_batch_gd",
            "sgd"
        ]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression():
    def __init__(self,  num_epochs = 1000, batch_size = 32,
        solver: Solvers = "sgd",
        learning_rate = 0.01):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.solver = solver
        self.batch_size = batch_size
    def fit(self, X, y):
        loss = BCE()
        if self.solver == "sgd":
            self.coef_, self.intercept_, self.loss_history_ = SGD(X, y, self.num_epochs, self.learning_rate, loss)
        elif self.solver == "batch_gd":
            self.coef_, self.intercept_, self.loss_history_ = Batch_GD(X, y, self.num_epochs, self.learning_rate, loss)
        elif self.solver == "mini_batch_gd":
            self.coef_, self.intercept_, self.loss_history_ = Mini_Batch_GD(X, y, self.num_epochs, self.learning_rate, self.batch_size, loss)
        else: raise ValueError(f"Solver {self.solver} is not supported")
        return self

    def predict(self, X):
        logits = X @ self.coef_ + self.intercept_
        return (logits >= 0).astype(np.int32)
    def score(self, X, y):
        ...

if __name__ == "__main__":
    train = np.loadtxt("/Volumes/DEV/OVTK/ML-fromscratch-sklearn/data/Logistic Regression Test/train.csv", delimiter=",", skiprows=1)
    test = np.loadtxt("/Volumes/DEV/OVTK/ML-fromscratch-sklearn/data/Logistic Regression Test/test.csv", delimiter=",", skiprows=1)

    X_train, y_train = train[:, :2], train[:, 2]
    X_test, y_test = test[:, :2], test[:, 2]

    model = LogisticRegression(
        solver="batch_gd",
        num_epochs=2000,
        learning_rate=0.1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    logits = X_test @ model.coef_ + model.intercept_
    test_loss = np.mean(
        np.logaddexp(0, logits) - y_test * logits
    )

    print("Weights:", model.coef_)
    print("Bias:", model.intercept_)
    print("Accuracy:", np.mean(y_pred == y_test))
    print("Test BCE:", test_loss)
