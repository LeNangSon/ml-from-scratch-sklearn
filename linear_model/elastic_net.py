from _solvers import Batch_GD, Mini_Batch_GD, SGD
from typing import Literal
from _losses import ElasticLoss
import numpy as np

Solvers = Literal[
        "batch_gd",
        "mini_batch_gd",
        "sgd"
]

class ElasticNet():
    def __init__(self,num_epochs = 1000, batch_size = 32 ,learning_rate = .01, solver : Solvers = "mini_batch_gd", alpha = .0, beta = .0):
        self.learning_rate = learning_rate
        self.solver = solver
        self.alpha = alpha
        self.beta = beta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
    def fit(self, X, y):

        loss = ElasticLoss()

        if (self.solver == "batch_gd"):
            self.coef_, self.intercept_ = Batch_GD(X, y, self.num_epochs, self.learning_rate, loss, self.alpha, self.beta)

        elif (self.solver == "mini_batch_gd"):
            self.coef_, self.intercept_ = Mini_Batch_GD(X, y, self.num_epochs, self.learning_rate, self.batch_size,loss, self.alpha, self.beta)

        elif (self.solver == "sgd"):
            self.coef_, self.intercept_ = SGD(X, y, self.num_epochs, self.learning_rate, loss, self.alpha, self.beta)

        else: raise(ValueError(f"Solver {self.solver} is not supported"))

        return self
    def score():
        ...

if __name__ == "__main__":
    model = ElasticNet(alpha = 1, beta = 1)
    
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