from scipy.signal import savgol_filter

def savgol(data, window_length, polyorder):
    """
    Apply Savitzky-Golay filter to intensity data.
    Parameters:
    data (np.array): Intensity data
    window_length (int): Length of the filter window
    polyorder (int): Order of the polynomial used to fit the samples

    Returns:
    smoothed_data (np.array): Intensity data after applying the Savitzky-Golay filter
    """
    return savgol_filter(data, window_length, polyorder)