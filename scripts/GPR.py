import pandas as pd
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def GPR(X, y):
    """
    Gaussian Process Regression (GPR) model
    Parameters:
    X (np.array): RT Data
    y (np.array): Intensity Data
    """
    X = X.reshape(-1, 1)

    kernel = ConstantKernel(1, (1e-2, 1e2)) * RBF(1, (1e-2, 1e2)) + WhiteKernel(1e-5, (1e-3, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha = 0)

    gpr.fit(X, y)

    X_pred = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    y_pred, sigma = gpr.predict(X_pred, return_std=True)

    return X_pred, y_pred, sigma