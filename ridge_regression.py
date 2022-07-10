import numpy as np
from scipy import linalg


class RidgeRegression:
    def __init__(self, lambda_ = 1):
        self.w = None
        self.lambda_ = lambda_
    
    def fit(self, X, t):
        Xtil = np.c_[np.ones(X.shape[0]),X]
        A = np.dot(Xtil.T, Xtil) + self.lambda_ * np.identity(Xtil.shape[1])
        b = np.dot(Xtil.T, t)
        self.w_ = np.linalg.solve(A,b)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape([1,-1])
        Xtil = np.c_[np.ones(X.shape[0]),X]
        return np.dot(Xtil, self.w_)

