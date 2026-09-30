import numpy as np

def closed_form(X, y):
    ones = np.ones(X.shape[0])
    X = np.c_[ones, X]
    return np.linalg.inv(X.T @ X) @ X.T @ y

def ridge_closed_form(X, y, alpha):
    ones = np.ones(X.shape[0])
    X = np.c_[ones, X]
    I = np.eye(X.shape[1])
    I[0,0] = 0           
    return np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y

    
    