import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def GPR(X, y, num_pred_points=500, downsample=10):
    """
    Gaussian Process Regression (GPR) model with options for downsampling and reduced prediction resolution.
    """
    if len(X) > downsample:
        X = X[::downsample]
        y = y[::downsample]
    
    X = X.reshape(-1, 1)
    
    kernel = ConstantKernel(1, (1e-2, 1e2)) * RBF(1, (1e-2, 1e2)) + WhiteKernel(1e-5, (1e-3, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10)
    
    gpr.fit(X, y)
    
    X_pred = np.linspace(X.min(), X.max(), num_pred_points).reshape(-1, 1)
    y_pred, sigma = gpr.predict(X_pred, return_std=True)
    
    return X_pred, y_pred, sigma
