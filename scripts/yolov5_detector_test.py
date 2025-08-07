#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for YOLOv5 Vehicle Detector
This script tests the functionality of the YOLOv5VehicleDetector class.
"""

import os
import cv2
import numpy as np
import tempfile
from distributed_image_processor import DistributedImageProcessor
from yolov5_vehicle_detector import YOLOv5VehicleDetector

def create_test_images(directory, num_images=5):
    """
    Create test images for testing the vehicle detector.
    
    Args:
        directory (str): Directory to save test images
        num_images (int): Number of test images to create
    """
    os.makedirs(directory, exist_ok=True)
    
    # Create sample images with different colors and shapes
    for i in range(num_images):
        # Create a blank image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some random shapes to simulate vehicles
        # Car 1 (rectangle)
        car_x = np.random.randint(50, 400)
        car_y = np.random.randint(200, 350)
        car_w = np.random.randint(80, 120)
        car_h = np.random.randint(40, 60)
        cv2.rectangle(img, (car_x, car_y), (car_x + car_w, car_y + car_h), (0, 0, 255), -1)
        
        # Car 2 (rectangle)
        car_x = np.random.randint(200, 500)
        car_y = np.random.randint(100, 250)
        car_w = np.random.randint(80, 120)
        car_h = np.random.randint(40, 60)
        cv2.rectangle(img, (car_x, car_y), (car_x + car_w, car_y + car_h), (255, 0, 0), -1)
        
        # Bus (larger rectangle)
        if i % 2 == 0:
            bus_x = np.random.randint(100, 300)
            bus_y = np.random.randint(300, 400)
            bus_w = np.random.randint(150, 200)
            bus_h = np.random.randint(60, 80)
            cv2.rectangle(img, (bus_x, bus_y), (bus_x + bus_w, bus_y + bus_h), (0, 255, 0), -1)
        
        # Add road markings
        cv2.line(img, (0, 400), (640, 400), (255, 255, 255), 2)
        cv2.line(img, (320, 0), (320, 480), (255, 255, 255), 2)
        
        # Save the image with a timestamp-like filename
        timestamp = f"20250417_{130000 + i * 100}"
        filename = f"traffic_{timestamp}.jpg"
        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, img)
        
        print(f"Created test image: {filepath}")

def test_yolov5_vehicle_detector():
    """Test the YOLOv5VehicleDetector functionality."""
    # Create a temporary directory for test images
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Create test images
        create_test_images(temp_dir)
        
        # Initialize processors
        img_processor = DistributedImageProcessor()
        
        # Load and preprocess images
        print("Loading and preprocessing images...")
        images_data = img_processor.load_images_from_directory(temp_dir)
        preprocessed_data = img_processor.preprocess_images(images_data)
        
        # Initialize vehicle detector
        print("Initializing YOLOv5 vehicle detector...")
        detector = YOLOv5VehicleDetector()
        
        try:
            # Detect vehicles
            print("Detecting vehicles...")
            detection_results = detector.detect_vehicles(preprocessed_data)
            
            # Count vehicles
            vehicle_counts = detector.count_vehicles_by_class(detection_results)
            print(f"Vehicle counts: {vehicle_counts}")
            
            # Visualize detections
            print("Visualizing detections...")
            vis_dir = os.path.join(temp_dir, "visualizations")
            vis_paths = detector.visualize_detections(detection_results, vis_dir)
            
            # Print detection results
            for i, result in enumerate(detection_results):
                print(f"\nImage {i+1}: {result['image_path']}")
                print(f"Inference time: {result.get('inference_time', 0):.3f}s")
                print(f"Detections: {len(result['detections'])}")
                for j, det in enumerate(result['detections']):
                    print(f"  {j+1}. {det['class_name']} ({det['confidence']:.2f})")
            
            print("\nTest completed successfully!")
            
        except Exception as e:
            print(f"Error in test: {e}")

if __name__ == "__main__":
    test_yolov5_vehicle_detector()
