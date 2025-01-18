import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

def pad_predictions(predictions, input_length, window_size):
    padding = [np.nan] * window_size
    padded_predictions = np.concatenate([padding, predictions.flatten()])
    return padded_predictions


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

    # Create sequences from the entire dataset
    X, y = create_sequence(normalized_y, window_size)

    # Reshape for RNN input
    X = X.reshape(X.shape[0], X.shape[1], 1)

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

    # Train the model on the entire dataset
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    # Predict on the entire dataset
    predictions = model.predict(X, verbose=1)

    # Inverse transform predictions
    predictions_inv = scaler.inverse_transform(predictions)
    predictions = pad_predictions(predictions_inv, len(normalized_y), window_size)

    return predictions
