from _losses import RidgeLoss
from _solvers import ridge_closed_form, SGD, Batch_GD, Mini_Batch_GD
import numpy as np
from typing import Literal

Solvers = Literal[
            "ridge_closed_form",
            "batch_gd",
            "mini_batch_gd",
            "sgd"
        ]


class RidgeRegression():
    def __init__(self, num_epochs = 1000, batch_size = 32, alpha = 1.0, learning_rate = .01, solver: Solvers = "ridge_closed_form"):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.solver = solver
        self.num_epochs = num_epochs
        self.batch_size = batch_size
    def fit(self, X, y):
        if self.solver == "ridge_closed_form":
            w = ridge_closed_form(X, y, self.alpha)
            self.coef_ = w[1:]
            self.intercept_ = w[0]
        else: 
            loss = RidgeLoss()
            if self.solver == "sgd":
                self.coef_, self.intercept_ = SGD(X, y, self.num_epochs, self.learning_rate, loss, self.alpha)
            elif self.solver == "batch_gd":
                self.coef_, self.intercept_ = Batch_GD(X, y, self.num_epochs, self.learning_rate, loss, self.alpha)
            elif self.solver == "mini_batch_gd":
                self.coef_, self.intercept_ = Mini_Batch_GD(X, y, self.num_epochs, self.learning_rate, self.batch_size, loss, self.alpha)
            else: raise ValueError(f"Solver {self.solver} is not supported")
        return self
    def score():
        ...

if __name__ == "__main__":
    model = RidgeRegression()

    X = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [2.5, 3.0],
    [3.0, 2.5],
    [4.0, 3.5],
    [5.0, 4.0],
    [6.0, 5.0],
    [7.0, 4.5]
    ])

    y = np.array([12.15, 10.96, 18.19, 17.96, 22.43, 25.93, 31.47, 31.73])

    model = model.fit(X,y)
    print(model.coef_)
    print(model.intercept_)