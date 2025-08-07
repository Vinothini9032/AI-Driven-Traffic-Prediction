#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySpark Distributed Image Processing for Traffic Congestion Model
This module implements distributed image processing using PySpark for the
AI-driven smart traffic congestion model.
"""

import os
import cv2
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType, StringType, StructType, StructField, ArrayType, FloatType

class PySparkImageProcessor:
    """
    A class for distributed image processing using PySpark.
    This class provides functionality to process traffic camera images in a distributed manner.
    """
    
    def __init__(self, master="local[*]", app_name="TrafficImageProcessor"):
        """
        Initialize the PySpark image processor.
        
        Args:
            master (str): Spark master URL
            app_name (str): Name of the Spark application
        """
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .getOrCreate()
        
        # Define schema for image data
        self.image_schema = StructType([
            StructField("image_path", StringType(), False),
            StructField("image_data", BinaryType(), True),
            StructField("width", FloatType(), True),
            StructField("height", FloatType(), True),
            StructField("timestamp", StringType(), True)
        ])
        
        # Register UDFs
        self._register_udfs()
    
    def _register_udfs(self):
        """Register user-defined functions for image processing."""
        
        # UDF to read image from binary data
        @udf(returnType=ArrayType(FloatType()))
        def decode_image(binary_data):
            """Convert binary data to numpy array."""
            if binary_data is None:
                return None
            nparr = np.frombuffer(binary_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            return img.tolist()
        
        # UDF to preprocess image for YOLOv5
        @udf(returnType=ArrayType(FloatType()))
        def preprocess_for_yolo(img_array, target_size=640):
            """Preprocess image for YOLOv5 inference."""
            if img_array is None:
                return None
            
            img = np.array(img_array, dtype=np.uint8)
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
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img.tolist()
        
        # Register UDFs
        self.spark.udf.register("decode_image", decode_image)
        self.spark.udf.register("preprocess_for_yolo", preprocess_for_yolo)
    
    def load_images_from_directory(self, directory_path):
        """
        Load images from a directory into a Spark DataFrame.
        
        Args:
            directory_path (str): Path to directory containing images
            
        Returns:
            DataFrame: Spark DataFrame containing image data
        """
        # List all image files in the directory
        image_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        # Create a DataFrame with image paths
        image_paths_df = self.spark.createDataFrame([(path,) for path in image_files], ["image_path"])
        
        # Define UDF to read image file
        @udf(returnType=BinaryType())
        def read_image_file(path):
            """Read image file and return binary data."""
            with open(path, "rb") as f:
                return f.read()
        
        # Define UDF to extract image dimensions
        @udf(returnType=StructType([
            StructField("width", FloatType(), True),
            StructField("height", FloatType(), True)
        ]))
        def get_image_dimensions(binary_data):
            """Extract image dimensions from binary data."""
            if binary_data is None:
                return None
            nparr = np.frombuffer(binary_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]
            return (float(w), float(h))
        
        # Register UDFs
        self.spark.udf.register("read_image_file", read_image_file)
        self.spark.udf.register("get_image_dimensions", get_image_dimensions)
        
        # Apply UDFs to read images and extract dimensions
        images_df = image_paths_df.withColumn("image_data", read_image_file("image_path"))
        images_df = images_df.withColumn("dimensions", get_image_dimensions("image_data"))
        images_df = images_df.withColumn("width", images_df.dimensions.width)
        images_df = images_df.withColumn("height", images_df.dimensions.height)
        images_df = images_df.drop("dimensions")
        
        # Extract timestamp from filename (assuming format contains timestamp)
        @udf(returnType=StringType())
        def extract_timestamp(path):
            """Extract timestamp from filename."""
            filename = os.path.basename(path)
            # This is a placeholder - adjust based on your actual filename format
            # Example: traffic_20250417_123045.jpg -> 2025-04-17 12:30:45
            parts = filename.split('_')
            if len(parts) >= 3:
                date_part = parts[1]
                time_part = parts[2].split('.')[0]
                if len(date_part) == 8 and len(time_part) == 6:
                    return f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
            return None
        
        # Register and apply timestamp extraction UDF
        self.spark.udf.register("extract_timestamp", extract_timestamp)
        images_df = images_df.withColumn("timestamp", extract_timestamp("image_path"))
        
        return images_df
    
    def preprocess_images(self, images_df):
        """
        Preprocess images for YOLOv5 inference.
        
        Args:
            images_df (DataFrame): Spark DataFrame containing image data
            
        Returns:
            DataFrame: DataFrame with preprocessed images
        """
        # Decode binary image data to array
        decoded_df = images_df.withColumn("decoded_image", self.spark.sql.functions.expr("decode_image(image_data)"))
        
        # Preprocess for YOLOv5
        preprocessed_df = decoded_df.withColumn("preprocessed_image", 
                                               self.spark.sql.functions.expr("preprocess_for_yolo(decoded_image)"))
        
        return preprocessed_df
    
    def batch_process(self, images_df, batch_size=16):
        """
        Process images in batches.
        
        Args:
            images_df (DataFrame): Spark DataFrame containing image data
            batch_size (int): Size of each batch
            
        Returns:
            list: List of batches, each containing preprocessed images
        """
        # Repartition to ensure proper distribution
        partitioned_df = images_df.repartition(self.spark.sparkContext.defaultParallelism)
        
        # Convert to RDD for batch processing
        images_rdd = partitioned_df.rdd
        
        # Group into batches
        batched_rdd = images_rdd.zipWithIndex().map(lambda x: (x[1] // batch_size, x[0])).groupByKey().map(lambda x: list(x[1]))
        
        return batched_rdd.collect()
    
    def stop(self):
        """Stop the Spark session."""
        self.spark.stop()


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = PySparkImageProcessor()
    
    # Example directory with traffic camera images
    image_dir = "/path/to/traffic/images"
    
    # Load and process images
    try:
        # Load images
        print("Loading images from directory...")
        images_df = processor.load_images_from_directory(image_dir)
        
        # Show DataFrame schema and sample
        print("DataFrame schema:")
        images_df.printSchema()
        
        print("Sample data:")
        images_df.show(5, truncate=False)
        
        # Preprocess images
        print("Preprocessing images...")
        preprocessed_df = processor.preprocess_images(images_df)
        
        # Process in batches
        print("Processing in batches...")
        batches = processor.batch_process(preprocessed_df)
        print(f"Created {len(batches)} batches")
        
    finally:
        # Stop Spark session
        processor.stop()
