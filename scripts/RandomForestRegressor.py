import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def RandomForestRegressor(X, y):
    """
    Random Forest Regression model for smoothing intensity data.
    Parameters:
    X (np.array): RT Data
    y (np.array): Intensity Data
    Returns:
    y_pred (np.array): Predicted intensity values
    """

    X = X.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, max_depth = 5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X)

    return y_pred