import numpy as np
from scipy.sparse import eye, diags
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

def pls(time, intensity, lambda_param=1E8, plot=False):
    """
    Apply penalized least squares smoothing to the input intensity data.

    Parameters:
    - time (np.array): Time series data (e.g., HPLC time)
    - intensity (np.array): Signal data corresponding to the time series
    - lambda_param (float): Regularization parameter to control smoothing (default=1E8)
    - plot (bool): If True, plots the original and smoothed signal (default=False)

    Returns:
    - smoothed_signal (np.array): Smoothed intensity values
    """
    # Length of the signal
    m = len(intensity)
    
    # Sparse identity matrix
    Espar = eye(m, format='csc')
    
    # Second-order difference matrix
    Dspar = diags([1, -2, 1], offsets=[0, 1, 2], shape=(m-2, m), format='csc')
    
    # Regularization matrix
    C = Espar + lambda_param * (Dspar.T @ Dspar)
    
    # Sparse Cholesky decomposition
    C_chol = splu(C)  # Sparse LU decomposition
    smoothed_signal = C_chol.solve(intensity)  # Solve for the smoothed signal
    
    # Optionally plot the result
    if plot:
        plt.figure()
        plt.plot(time, intensity, 'g', label='Original Signal')
        plt.plot(time, smoothed_signal, 'r', linewidth=1.5, label='Smoothed Signal')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.title('Penalized Least Squares Smoothing')
        plt.show()
    
    return smoothed_signal

# Example usage:
# time = np.array([...])       # Replace with your time data
# intensity = np.array([...]) # Replace with your intensity data
# smoothed_signal = penalized_least_squares_smoothing(time, intensity, lambda_param=1E8, plot=True)
