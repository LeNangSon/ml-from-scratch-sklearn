from typing import Literal
from _solvers import closed_form
import numpy as np

class LinearRegression():
    def __init__(self, 
            solvers: Literal[
            "closed_form",
            "batch_gd",
            "mini_batch_gd",
            "sgd"
        ] = "closed_form",
        learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.solvers = solvers
    def fit(self, X, y):
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0
        ones = np.ones(X.shape[0])
        X_ones = np.c_[ones, X]
        if self.solvers == "closed_form":
            w = closed_form(X_ones, y)
        elif self.solvers == "batch_gd":
            print("ok")
        elif self.solvers == "batch_gd":
            ...
        elif self.solvers == "mini_batch_gd":
            ...
        elif self.solvers == "sgd":
            ...
        else: print(f"Your solver {self.solvers} is not supported")
        self.coef_ = w[1:]
        self.intercept_ = w[0]
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def score(self):
        ...

    

