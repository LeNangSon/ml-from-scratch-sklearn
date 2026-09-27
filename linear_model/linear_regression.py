from typing import Literal
from _solvers import closed_form
from _solvers import SGD, Batch_GD, Mini_Batch_GD
from _losses import MSE
import numpy as np


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
                self.coef_, self.intercept_ = SGD(X, y, self.num_epochs, self.learning_rate, loss)
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

    

