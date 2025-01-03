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

    length_scale = (X.max() - X.min()) / 10
    kernel = (ConstantKernel(1, (1e-2, 1e2)) *
    RBF(length_scale, (1e-2, 1e2)) +
    WhiteKernel(1e-5, (1e-3, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha = 1e-10)

    gpr.fit(X, y)

    X_pred = np.linspace(X.min() - 0.1 * (X.max() - X.min()), X.max() + 0.1 * (X.max() - X.min()), 1000).reshape(-1, 1)
    y_pred, sigma = gpr.predict(X_pred, return_std=True)

    return X_pred, y_pred, sigma