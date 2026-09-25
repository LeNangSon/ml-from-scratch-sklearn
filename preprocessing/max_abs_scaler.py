import numpy as np

class MaxAbsScaler():
    def fit(self, X):
        self.scale_ = np.max(np.abs(X))
        return self
    def transform(self, X):
        return X / self.scale_
    def fit_transform(self, X):
        return self.fit(X).transform(X)