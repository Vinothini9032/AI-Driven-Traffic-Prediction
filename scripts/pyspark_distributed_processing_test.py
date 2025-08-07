#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for PySpark Distributed Image Processing
This script tests the functionality of the PySparkImageProcessor class.
"""

import os
import cv2
import numpy as np
import tempfile
from pyspark_image_processor import PySparkImageProcessor

def create_test_images(directory, num_images=5):
    """
    Create test images for testing the PySpark image processor.
    
    Args:
        directory (str): Directory to save test images
        num_images (int): Number of test images to create
    """
    os.makedirs(directory, exist_ok=True)
    
    # Create sample images with different colors and shapes
    for i in range(num_images):
        # Create a blank image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some random shapes
        color = tuple(np.random.randint(0, 255, 3).tolist())
        center = (np.random.randint(100, 540), np.random.randint(100, 380))
        radius = np.random.randint(30, 100)
        
        # Draw a circle
        cv2.circle(img, center, radius, color, -1)
        
        # Add a rectangle representing a vehicle
        rect_x = np.random.randint(50, 500)
        rect_y = np.random.randint(50, 350)
        rect_w = np.random.randint(50, 150)
        rect_h = np.random.randint(30, 80)
        rect_color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), rect_color, -1)
        
        # Save the image with a timestamp-like filename
        timestamp = f"20250417_{130000 + i * 100}"
        filename = f"traffic_{timestamp}.jpg"
        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, img)
        
        print(f"Created test image: {filepath}")

def test_pyspark_image_processor():
    """Test the PySparkImageProcessor functionality."""
    # Create a temporary directory for test images
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Create test images
        create_test_images(temp_dir)
        
        # Initialize the processor
        processor = PySparkImageProcessor(app_name="TestImageProcessor")
        
        try:
            # Load images
            print("Loading images from directory...")
            images_df = processor.load_images_from_directory(temp_dir)
            
            # Show DataFrame schema and sample
            print("DataFrame schema:")
            images_df.printSchema()
            
            print("Sample data:")
            images_df.show(5, truncate=False)
            
            # Count the number of images
            image_count = images_df.count()
            print(f"Loaded {image_count} images")
            
            # Preprocess images
            print("Preprocessing images...")
            preprocessed_df = processor.preprocess_images(images_df)
            
            # Process in batches
            print("Processing in batches...")
            batches = processor.batch_process(preprocessed_df, batch_size=2)
            print(f"Created {len(batches)} batches")
            
            print("Test completed successfully!")
            
        finally:
            # Stop Spark session
            processor.stop()

if __name__ == "__main__":
    test_pyspark_image_processor()
