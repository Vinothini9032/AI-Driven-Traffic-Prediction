#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RNN-LSTM Traffic Volume Prediction Model
This module implements a Recurrent Neural Network with Long Short-Term Memory (RNN-LSTM)
for traffic volume prediction in the AI-driven smart traffic congestion model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class TrafficVolumePredictor:
    """
    A class for traffic volume prediction using RNN-LSTM.
    This class provides functionality to predict traffic volume for the coming hours.
    """
    
    def __init__(self, sequence_length=24, prediction_horizon=12, features=None):
        """
        Initialize the traffic volume predictor.
        
        Args:
            sequence_length (int): Number of time steps to use for prediction
            prediction_horizon (int): Number of time steps to predict ahead
            features (list): List of feature names to use for prediction
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.features = features or ['volume', 'hour', 'day_of_week', 'is_weekend', 'is_holiday']
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
        # Set random seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def _create_dataset(self, data, time_column='timestamp'):
        """
        Create a dataset for LSTM training from time series data.
        
        Args:
            data (DataFrame): DataFrame containing traffic volume data
            time_column (str): Name of the column containing timestamps
            
        Returns:
            tuple: X (input sequences), y (target values), and timestamps
        """
        # Ensure data is sorted by time
        data = data.sort_values(by=time_column)
        
        # Extract features
        features_data = data[self.features].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features_data)
        
        X, y, timestamps = [], [], []
        
        # Create sequences
        for i in range(len(scaled_features) - self.sequence_length - self.prediction_horizon + 1):
            X.append(scaled_features[i:i + self.sequence_length])
            y.append(scaled_features[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, 0])  # Volume is the first feature
            timestamps.append(data[time_column].iloc[i + self.sequence_length])
        
        return np.array(X), np.array(y), np.array(timestamps)
    
    def _build_model(self, input_shape):
        """
        Build the RNN-LSTM model.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, num_features)
            
        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=32),
            Dropout(0.2),
            Dense(self.prediction_horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, data, time_column='timestamp', validation_split=0.2, epochs=50, batch_size=32, verbose=1):
        """
        Train the RNN-LSTM model.
        
        Args:
            data (DataFrame): DataFrame containing traffic volume data
            time_column (str): Name of the column containing timestamps
            validation_split (float): Fraction of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level
            
        Returns:
            History: Training history
        """
        # Create dataset
        X, y, _ = self._create_dataset(data, time_column)
        
        # Build model
        self.model = self._build_model((X.shape[1], X.shape[2]))
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        # Train model
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, data, time_column='timestamp'):
        """
        Predict traffic volume for the coming hours.
        
        Args:
            data (DataFrame): DataFrame containing recent traffic volume data
            time_column (str): Name of the column containing timestamps
            
        Returns:
            DataFrame: DataFrame with predictions and timestamps
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create dataset
        X, _, timestamps = self._create_dataset(data, time_column)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Inverse transform predictions
        inverse_predictions = np.zeros((predictions.shape[0], predictions.shape[1], len(self.features)))
        inverse_predictions[:, :, 0] = predictions  # Volume is the first feature
        
        # Reshape for inverse transform
        inverse_predictions = inverse_predictions.reshape(-1, len(self.features))
        inverse_predictions = self.scaler.inverse_transform(inverse_predictions)
        
        # Extract volume predictions
        volume_predictions = inverse_predictions[:, 0].reshape(-1, self.prediction_horizon)
        
        # Create result DataFrame
        result = []
        for i, timestamp in enumerate(timestamps):
            base_time = pd.to_datetime(timestamp)
            for j in range(self.prediction_horizon):
                pred_time = base_time + timedelta(hours=j+1)
                result.append({
                    'timestamp': pred_time,
                    'predicted_volume': volume_predictions[i, j]
                })
        
        return pd.DataFrame(result)
    
    def evaluate(self, data, time_column='timestamp'):
        """
        Evaluate the model on test data.
        
        Args:
            data (DataFrame): DataFrame containing test data
            time_column (str): Name of the column containing timestamps
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create dataset
        X, y_true, _ = self._create_dataset(data, time_column)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    
    def plot_history(self):
        """
        Plot training history.
        
        Returns:
            tuple: Figure and axes
        """
        if self.history is None:
            raise ValueError("Model not trained. Call train() first.")
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax[0].plot(self.history.history['loss'], label='Training Loss')
        ax[0].plot(self.history.history['val_loss'], label='Validation Loss')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        
        # Plot MAE
        ax[1].plot(self.history.history['mae'], label='Training MAE')
        ax[1].plot(self.history.history['val_mae'], label='Validation MAE')
        ax[1].set_title('Mean Absolute Error')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('MAE')
        ax[1].legend()
        
        plt.tight_layout()
        return fig, ax
    
    def plot_predictions(self, predictions, actual=None, num_samples=5):
        """
        Plot predictions against actual values.
        
        Args:
            predictions (DataFrame): DataFrame with predictions
            actual (DataFrame): DataFrame with actual values
            num_samples (int): Number of time points to plot
            
        Returns:
            tuple: Figure and axes
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group predictions by base timestamp
        grouped = predictions.groupby('base_timestamp')
        
        # Plot a few samples
        for i, (timestamp, group) in enumerate(grouped.head(num_samples).items()):
            times = pd.to_datetime(group['timestamp'])
            ax.plot(times, group['predicted_volume'], 'o-', label=f'Prediction {i+1}')
            
            if actual is not None:
                # Find actual values for the same time period
                mask = (actual['timestamp'] >= times.min()) & (actual['timestamp'] <= times.max())
                if mask.any():
                    actual_subset = actual[mask]
                    ax.plot(actual_subset['timestamp'], actual_subset['volume'], 'x--', 
                           label=f'Actual {i+1}')
        
        ax.set_title('Traffic Volume Predictions')
        ax.set_xlabel('Time')
        ax.set_ylabel('Volume')
        ax.legend()
        
        plt.tight_layout()
        return fig, ax
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.save(filepath)
        
        # Save scaler
        scaler_path = os.path.splitext(filepath)[0] + '_scaler.npy'
        np.save(scaler_path, self.scaler)
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'features': self.features
        }
        metadata_path = os.path.splitext(filepath)[0] + '_metadata.npy'
        np.save(metadata_path, metadata)
    
    def load_model(self, filepath):
        """
        Load the model from a file.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        
        # Load scaler
        scaler_path = os.path.splitext(filepath)[0] + '_scaler.npy'
        self.scaler = np.load(scaler_path, allow_pickle=True).item()
        
        # Load metadata
        metadata_path = os.path.splitext(filepath)[0] + '_metadata.npy'
        metadata = np.load(metadata_path, allow_pickle=True).item()
        
        self.sequence_length = metadata['sequence_length']
        self.prediction_horizon = metadata['prediction_horizon']
        self.features = metadata['features']


def generate_synthetic_traffic_data(num_days=30, start_date=None):
    """
    Generate synthetic traffic volume data for testing.
    
    Args:
        num_days (int): Number of days to generate data for
        start_date (str): Start date in format 'YYYY-MM-DD'
        
    Returns:
        DataFrame: DataFrame with synthetic traffic data
    """
    if start_date is None:
        start_date = '2025-01-01'
    
    # Create timestamp range (hourly data)
    timestamps = pd.date_range(start=start_date, periods=num_days*24, freq='H')
    
    # Initialize data
    data = []
    
    # Base volume patterns
    hourly_pattern = np.array([
        0.2, 0.1, 0.05, 0.05, 0.1, 0.3,  # 0-5 AM
        0.6, 0.9, 1.0, 0.8, 0.7, 0.8,    # 6-11 AM
        0.9, 0.8, 0.7, 0.8, 0.9, 1.0,    # 12-5 PM
        0.8, 0.6, 0.5, 0.4, 0.3, 0.2     # 6-11 PM
    ])
    
    # Day of week factors (Monday=0, Sunday=6)
    day_factors = np.array([1.0, 1.0, 1.0, 1.0, 1.1, 0.8, 0.7])
    
    # Generate data
    for timestamp in timestamps:
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Determine if it's a holiday (simplified)
        is_holiday = 1 if (timestamp.month == 1 and timestamp.day == 1) or \
                          (timestamp.month == 12 and timestamp.day == 25) else 0
        
        # Calculate base volume
        base_volume = 100 * hourly_pattern[hour] * day_factors[day_of_week]
        
        # Add random variation
        noise = np.random.normal(0, 0.1)
        volume = max(0, base_volume * (1 + noise))
        
        # Add seasonal component
        seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * timestamp.dayofyear / 365)
        volume *= seasonal_factor
        
        # Round to integer
        volume = int(volume)
        
        data.append({
            'timestamp': timestamp,
            'volume': volume,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday
        })
    
    return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    data = generate_synthetic_traffic_data(num_days=60)
    
    # Split into train and test
    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]
    
    # Initialize predictor
    predictor = TrafficVolumePredictor(sequence_length=24, prediction_horizon=12)
    
    # Train model
    history = predictor.train(train_data, epochs=20, verbose=1)
    
    # Evaluate model
    metrics = predictor.evaluate(test_data)
    print(f"Evaluation metrics: {metrics}")
    
    # Make predictions
    predictions = predictor.predict(test_data.head(24))
    print(predictions.head())
    
    # Plot results
    predictor.plot_history()
    plt.savefig('training_history.png')
    
    # Save model
    predictor.save_model('traffic_volume_predictor.h5')
