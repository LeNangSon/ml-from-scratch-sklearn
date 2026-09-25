# range [a, b] 'll be implemented later
import numpy as np

class MinMaxScaler():
    def fit(self, X):
        self.min_ = np.min(X, axis = 0)
        self.scale_ = np.max(X, axis = 0) - self.min_
        return self
    def transform(self, X):
        return (X - self.min_) / self.scale_
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    