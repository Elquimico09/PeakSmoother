import pandas as pd
import numpy as np
from scipy.sparse import diags, identity
from scipy.linalg import solve

def PenalizedLeastSquare(x, y, lambda_, p_order = 2):
    """
    Apply Penalized Least Square smoothing to intensity data.

    Parameters:
    x (np.array): RT Data
    y (np.array): Intensity Data
    lambda_ (float): Penalty parameter
    p_order (int): Order of the polynomial used to fit the samples
    Returns:
    smoothed_data (np.array): Intensity data after applying the Penalized Least Square smoothing
    """

    n = len(y)
    I = np.eye(n)

    D = diags([1, -2, 1], [0, 1, 2], shape=(n-2, n)).toarray()
    P = np.dot(D.T, D)

    # Solve (I + lambda * P) * y_smooth = y
    y_smooth = solve(I + lambda_ * P, y)
    return y_smooth