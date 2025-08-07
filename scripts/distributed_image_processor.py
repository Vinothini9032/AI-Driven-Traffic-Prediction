#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Distributed Image Processing for Traffic Congestion Model
This module implements a modified version of distributed image processing
that can work within the sandbox environment constraints.
"""

import os
import cv2
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

class DistributedImageProcessor:
    """
    A class for distributed image processing using multiprocessing.
    This class provides functionality to process traffic camera images in a parallel manner.
    """
    
    def __init__(self, num_workers=None):
        """
        Initialize the distributed image processor.
        
        Args:
            num_workers (int): Number of worker processes to use
        """
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        print(f"Initializing distributed image processor with {self.num_workers} workers")
    
    def load_images_from_directory(self, directory_path):
        """
        Load images from a directory.
        
        Args:
            directory_path (str): Path to directory containing images
            
        Returns:
            list: List of dictionaries containing image data
        """
        # List all image files in the directory
        image_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        print(f"Found {len(image_files)} images in {directory_path}")
        
        # Process images in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks
            future_to_path = {executor.submit(self._load_single_image, path): path for path in image_files}
            
            # Collect results
            images_data = []
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        images_data.append(result)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
        
        return images_data
    
    def _load_single_image(self, image_path):
        """
        Load a single image and extract its metadata.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary containing image data and metadata
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                return None
            
            # Extract dimensions
            height, width = img.shape[:2]
            
            # Extract timestamp from filename (assuming format contains timestamp)
            filename = os.path.basename(image_path)
            timestamp = self._extract_timestamp(filename)
            
            return {
                "image_path": image_path,
                "image": img,
                "width": width,
                "height": height,
                "timestamp": timestamp
            }
        except Exception as e:
            print(f"Error in _load_single_image for {image_path}: {e}")
            return None
    
    def _extract_timestamp(self, filename):
        """
        Extract timestamp from filename.
        
        Args:
            filename (str): Image filename
            
        Returns:
            str: Extracted timestamp or None
        """
        # This is a placeholder - adjust based on your actual filename format
        # Example: traffic_20250417_123045.jpg -> 2025-04-17 12:30:45
        parts = filename.split('_')
        if len(parts) >= 3:
            date_part = parts[1]
            time_part = parts[2].split('.')[0]
            if len(date_part) == 8 and len(time_part) == 6:
                return f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def preprocess_images(self, images_data, target_size=640):
        """
        Preprocess images for YOLOv5 inference.
        
        Args:
            images_data (list): List of dictionaries containing image data
            target_size (int): Target size for preprocessing
            
        Returns:
            list: List of dictionaries with preprocessed images
        """
        # Process images in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks
            future_to_idx = {executor.submit(self._preprocess_single_image, img_data, target_size): i 
                            for i, img_data in enumerate(images_data)}
            
            # Collect results
            results = [None] * len(images_data)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error preprocessing image {idx}: {e}")
                    results[idx] = images_data[idx]  # Keep original on error
        
        return results
    
    def _preprocess_single_image(self, img_data, target_size=640):
        """
        Preprocess a single image for YOLOv5 inference.
        
        Args:
            img_data (dict): Dictionary containing image data
            target_size (int): Target size for preprocessing
            
        Returns:
            dict: Dictionary with preprocessed image
        """
        try:
            img = img_data["image"]
            h, w = img.shape[:2]
            
            # Calculate resize ratio
            r = target_size / max(h, w)
            if r != 1:
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=interp)
            
            # Pad to square
            new_h, new_w = img.shape[:2]
            pad_h, pad_w = target_size - new_h, target_size - new_w
            pad_h, pad_w = max(0, pad_h), max(0, pad_w)
            
            # Divide padding into top, bottom, left, right
            top, bottom = pad_h // 2, pad_h - (pad_h // 2)
            left, right = pad_w // 2, pad_w - (pad_w // 2)
            
            # Apply padding
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            
            # Convert to RGB (YOLOv5 expects RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create a copy of the original data
            result = img_data.copy()
            
            # Update with preprocessed image
            result["preprocessed_image"] = img
            result["pad_info"] = (top, bottom, left, right)
            result["resize_ratio"] = r
            
            return result
        except Exception as e:
            print(f"Error in _preprocess_single_image: {e}")
            return img_data
    
    def batch_process(self, images_data, batch_size=16):
        """
        Process images in batches.
        
        Args:
            images_data (list): List of dictionaries containing image data
            batch_size (int): Size of each batch
            
        Returns:
            list: List of batches, each containing preprocessed images
        """
        # Create batches
        batches = []
        for i in range(0, len(images_data), batch_size):
            batch = images_data[i:i + batch_size]
            batches.append(batch)
        
        return batches


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = DistributedImageProcessor()
    
    # Example directory with traffic camera images
    image_dir = "/path/to/traffic/images"
    
    # Load and process images
    try:
        # Load images
        print("Loading images from directory...")
        images_data = processor.load_images_from_directory(image_dir)
        
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
        batches = processor.batch_process(preprocessed_data)
        print(f"Created {len(batches)} batches")
        
    except Exception as e:
        print(f"Error in main: {e}")
