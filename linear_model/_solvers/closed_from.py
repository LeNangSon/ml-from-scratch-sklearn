import numpy as np

def closed_form(X, y):
    return np.linalg.inv(X.T @ X) @ X @ y