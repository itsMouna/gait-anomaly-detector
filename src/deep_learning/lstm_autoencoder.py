import numpy as np


class LSTMAutoencoder:
    """
    LSTM Autoencoder for time-series anomaly detection.

    Trains to reconstruct normal windows. At inference,
    windows with high reconstruction error are flagged as anomalies.
    """

    def __init__(self, timesteps: int = 128, features: int = 1):
        self.timesteps = timesteps
        self.features = features
        self.model = self._build_model()

    def _build_model(self):
        # Lazy import so the module can be imported without TensorFlow
        # if only classical models are needed.
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
        )
        from tensorflow.keras.optimizers import Adam

        inputs = Input(shape=(self.timesteps, self.features))

        # Encoder
        x = LSTM(64, activation="tanh", return_sequences=False)(inputs)
        x = Dropout(0.2)(x)

        # Bottleneck
        bottleneck = RepeatVector(self.timesteps)(x)

        # Decoder
        x = LSTM(64, activation="tanh", return_sequences=True)(bottleneck)
        x = Dropout(0.2)(x)
        outputs = TimeDistributed(Dense(self.features))(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        return model

    def fit(self, X_train: np.ndarray, epochs: int = 15, batch_size: int = 32):
        """Train on normal windows (X_train → X_train reconstruction)."""
        self.model.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
        )
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Return per-window mean squared reconstruction error."""
        reconstructed = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructed, 2), axis=(1, 2))
        return mse

    def score(self, X: np.ndarray) -> np.ndarray:
        """Alias for reconstruction_error (higher = more anomalous)."""
        return self.reconstruction_error(X)
