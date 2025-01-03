import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

def RNN(y, window_size=30, epochs=100, batch_size=32, learning_rate=0.0001):
    """
    Recurrent Neural Network (RNN) model for smoothing intensity data.

    Parameters:
    y (np.array): Intensity data
    window_size (int): Number of time steps in each sequence
    epochs (int): Number of training epochs
    batch_size (int): Size of training batches
    learning_rate (float): Learning rate for the optimizer

    Returns:
    y_pred (np.array): Predicted intensity values (smoothed)
    """
    # Normalize y
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_y = scaler.fit_transform(y.reshape(-1, 1))

    # Create sequences
    def create_sequence(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)

    # Split into training and testing sets
    train_data, test_data = train_test_split(normalized_y, test_size=0.2, shuffle=False)
    X_train, y_train = create_sequence(train_data, window_size)
    X_test, y_test = create_sequence(test_data, window_size)

    # Reshape for RNN input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build RNN model
    model = Sequential([
        LSTM(128, activation="tanh", input_shape=(window_size, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation="tanh", return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation="tanh", return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test))

    # Predict on the test set
    predictions = model.predict(X_test, verbose=1)

    # Inverse transform predictions
    predictions_inv = scaler.inverse_transform(predictions)

    return predictions_inv
