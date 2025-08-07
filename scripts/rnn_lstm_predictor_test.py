#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for RNN-LSTM Traffic Volume Predictor
This script tests the functionality of the TrafficVolumePredictor class.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rnn_lstm_predictor import TrafficVolumePredictor, generate_synthetic_traffic_data

def test_rnn_lstm_predictor():
    """Test the RNN-LSTM Traffic Volume Predictor functionality."""
    print("Testing RNN-LSTM Traffic Volume Predictor...")
    
    # Create output directory for results
    output_dir = "/home/ubuntu/traffic_congestion_model/output/lstm_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data
    print("Generating synthetic traffic data...")
    data = generate_synthetic_traffic_data(num_days=30)
    
    # Save sample data
    data.head(100).to_csv(f"{output_dir}/sample_traffic_data.csv", index=False)
    print(f"Sample data saved to {output_dir}/sample_traffic_data.csv")
    
    # Plot sample data
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['volume'])
    plt.title('Synthetic Traffic Volume Data')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/synthetic_data_plot.png")
    print(f"Sample data plot saved to {output_dir}/synthetic_data_plot.png")
    
    # Split into train and test
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Initialize predictor with smaller parameters for faster testing
    print("Initializing RNN-LSTM predictor...")
    predictor = TrafficVolumePredictor(sequence_length=12, prediction_horizon=6)
    
    try:
        # Train model with fewer epochs for testing
        print("Training model (with reduced epochs for testing)...")
        history = predictor.train(train_data, epochs=5, batch_size=32, verbose=1)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = predictor.evaluate(test_data)
        print(f"Evaluation metrics: MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
        
        # Make predictions
        print("Making predictions...")
        predictions = predictor.predict(test_data.head(24))
        
        # Save predictions
        predictions.to_csv(f"{output_dir}/predictions.csv", index=False)
        print(f"Predictions saved to {output_dir}/predictions.csv")
        
        # Plot training history
        print("Plotting training history...")
        fig, ax = predictor.plot_history()
        plt.savefig(f"{output_dir}/training_history.png")
        plt.close(fig)
        print(f"Training history plot saved to {output_dir}/training_history.png")
        
        # Add base_timestamp to predictions for plotting
        predictions['base_timestamp'] = predictions['timestamp'].apply(
            lambda x: pd.to_datetime(x) - pd.Timedelta(hours=1))
        
        # Plot a few prediction samples
        print("Plotting prediction samples...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group predictions by base timestamp
        grouped = predictions.groupby('base_timestamp')
        
        # Plot a few samples
        for i, (timestamp, group) in enumerate(list(grouped)[:3]):
            times = pd.to_datetime(group['timestamp'])
            ax.plot(times, group['predicted_volume'], 'o-', label=f'Prediction {i+1}')
        
        ax.set_title('Traffic Volume Predictions')
        ax.set_xlabel('Time')
        ax.set_ylabel('Volume')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_samples.png")
        plt.close(fig)
        print(f"Prediction samples plot saved to {output_dir}/prediction_samples.png")
        
        # Save model
        model_path = f"{output_dir}/traffic_volume_predictor.h5"
        predictor.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in test: {e}")
        return False

if __name__ == "__main__":
    test_rnn_lstm_predictor()
