from typing import Literal
from _solvers import closed_form, SGD, Batch_GD, Mini_Batch_GD
from _losses import MSE
import numpy as np
import matplotlib.pyplot as plt

Solvers = Literal[
            "closed_form",
            "batch_gd",
            "mini_batch_gd",
            "sgd"
        ]

class LinearRegression():
    def __init__(self,  num_epochs = 1000, batch_size = 32,
        solver: Solvers = "closed_form",
        learning_rate = 0.01):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.solver = solver
        self.batch_size = batch_size
    def fit(self, X, y):
        if self.solver == "closed_form":
            w = closed_form(X, y)
            self.coef_ = w[1:]
            self.intercept_ = w[0]
        else: 
            loss = MSE()
            if self.solver == "sgd":
                self.coef_, self.intercept_, self.loss_history_ = SGD(X, y, self.num_epochs, self.learning_rate, loss)
            elif self.solver == "batch_gd":
                self.coef_, self.intercept_ = Batch_GD(X, y, self.num_epochs, self.learning_rate, loss)
            elif self.solver == "mini_batch_gd":
                self.coef_, self.intercept_ = Mini_Batch_GD(X, y, self.num_epochs, self.learning_rate, self.batch_size, loss)
            else: raise ValueError(f"Solver {self.solver} is not supported")
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        ...

if __name__ == "__main__":

    X = np.array([
    [1.0],
    [2.0],
    [2.5],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0]
    ])
    y = np.array([12.15, 10.96, 18.19, 17.96, 22.43, 25.93, 31.47, 31.73])

    learning_rates = [0.001, 0.003, 0.01, 0.03]

    fig, axes = plt.subplots(2 , 2, sharex=True, sharey=True, figsize = (8,8))

    axes = axes.flatten()

    for ax, learning_rate in zip(axes, learning_rates):
        num_epochs = 100
        model = LinearRegression(solver="sgd", learning_rate=learning_rate, num_epochs=num_epochs)
        model = model.fit(X, y)

        ax.plot(range(num_epochs * X.shape[0]), model.loss_history_)
        ax.set_title(f"lr: {learning_rate}\n"
                      f"final loss: {model.loss_history_[-1]:.4f}")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
    plt.tight_layout()
    plt.show()