#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Distributed Image Processing
This script tests the functionality of the DistributedImageProcessor class.
"""

import os
import cv2
import numpy as np
import tempfile
from distributed_image_processor import DistributedImageProcessor

def create_test_images(directory, num_images=5):
    """
    Create test images for testing the distributed image processor.
    
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

def test_distributed_image_processor():
    """Test the DistributedImageProcessor functionality."""
    # Create a temporary directory for test images
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Create test images
        create_test_images(temp_dir)
        
        # Initialize the processor
        processor = DistributedImageProcessor()
        
        try:
            # Load images
            print("Loading images from directory...")
            images_data = processor.load_images_from_directory(temp_dir)
            
            # Show sample data
            print(f"Loaded {len(images_data)} images")
            if images_data:
                sample = images_data[0]
                print(f"Sample image: {sample['image_path']}")
                print(f"Dimensions: {sample['width']}x{sample['height']}")
                print(f"Timestamp: {sample['timestamp']}")
            
            # Preprocess images
            print("Preprocessing images...")
            preprocessed_data = processor.preprocess_images(images_data)
            
            # Process in batches
            print("Processing in batches...")
            batches = processor.batch_process(preprocessed_data, batch_size=2)
            print(f"Created {len(batches)} batches")
            
            print("Test completed successfully!")
            
        except Exception as e:
            print(f"Error in test: {e}")

if __name__ == "__main__":
    test_distributed_image_processor()
