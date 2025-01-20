import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.cluster import KMeans

def downsample_with_kmeans(X, y, n_clusters=500):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X.reshape(-1, 1))
    cluster_centers = kmeans.cluster_centers_
    cluster_indices = [np.argmin(np.abs(X - center)) for center in cluster_centers]
    return X[cluster_indices], y[cluster_indices]

def GPR(X, y, num_pred_points=15000, downsample=3, use_kmeans=True):
    if len(X) > downsample:
        if use_kmeans:
            X, y = downsample_with_kmeans(X, y, n_clusters=len(X)//downsample * 2)
        else:
            X = X[::downsample]
            y = y[::downsample]

    X = X.reshape(-1, 1)

    kernel = ConstantKernel(1, (1e-2, 1e2)) * RBF(10, (1e-1, 1e3)) + WhiteKernel(1e-6, (1e-6, 1e0))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-10)
    gpr.fit(X, y)

    num_pred_points = min(len(X), num_pred_points)
    X_pred = np.linspace(X.min(), X.max(), num_pred_points).reshape(-1, 1)
    y_pred, sigma = gpr.predict(X_pred, return_std=True)

    return X_pred, y_pred, sigma
