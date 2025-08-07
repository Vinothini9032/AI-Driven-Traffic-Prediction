#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the AI-Driven Smart Traffic Congestion Model
This script provides a unified interface to run the complete system.
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Import our custom modules
from scripts.distributed_image_processor import DistributedImageProcessor
from scripts.distributed_image_processor import DistributedImageProcessor

from scripts.yolov5_vehicle_detector import YOLOv5VehicleDetector
from scripts.rnn_lstm_predictor import TrafficVolumePredictor
from scripts.fuzzy_logic_controller import FuzzyTrafficController
from scripts.carla_simulation_integration import CARLASimulationIntegration
from scripts.model_tester import ModelTester

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI-Driven Smart Traffic Congestion Model')
    parser.add_argument('--mode', type=str, default='simulation',
                        choices=['simulation', 'test', 'component'],
                        help='Operation mode: simulation, test, or component')
    parser.add_argument('--component', type=str, default=None,
                        choices=['processor', 'detector', 'predictor', 'controller'],
                        help='Component to run in component mode')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for results')
    parser.add_argument('--duration', type=int, default=3600,
                        help='Simulation duration in seconds')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive control in simulation mode')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run in selected mode
    if args.mode == 'simulation':
        run_simulation(args)
    elif args.mode == 'test':
        run_tests(args)
    elif args.mode == 'component':
        run_component(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

def run_simulation(args):
    """Run the CARLA simulation."""
    print("Starting AI-Driven Smart Traffic Congestion Model Simulation...")
    print(f"Output directory: {args.output}")
    print(f"Simulation duration: {args.duration} seconds")
    print(f"Adaptive control: {'Enabled' if args.adaptive else 'Disabled'}")
    
    # Create simulation
    simulation = CARLASimulationIntegration(output_dir=args.output)
    
    # Initialize components
    simulation.initialize_components()
    
    # Set up simulation
    simulation.setup_simulation()
    
    # Configure adaptive control
    if not args.adaptive:
        # Override the optimize_traffic_signals method to use fixed timing
        def fixed_timing(self):
            print(f"[{self._get_simulation_time_str()}] Using fixed signal timing...")
            
            # Apply fixed timing to all intersections
            for intersection_id, intersection in self.intersections.items():
                # Fixed cycle length of 120 seconds
                cycle_length = 120
                
                # Fixed green times (60 seconds for each direction)
                green_time_ns = 55
                green_time_ew = 55
                
                # Fixed yellow and all-red times
                yellow_time = 3
                all_red_time = 2
                
                # Update traffic lights
                ns_light = self.traffic_lights[f"{intersection_id}_NS"]
                ew_light = self.traffic_lights[f"{intersection_id}_EW"]
                
                ns_light["cycle_length"] = cycle_length
                ns_light["green_time"] = green_time_ns
                ns_light["yellow_time"] = yellow_time
                ns_light["all_red_time"] = all_red_time
                
                ew_light["cycle_length"] = cycle_length
                ew_light["green_time"] = green_time_ew
                ew_light["yellow_time"] = yellow_time
                ew_light["all_red_time"] = all_red_time
                
                print(f"  Intersection {intersection_id}: Fixed cycle length of {cycle_length}s")
                print(f"    N-S: Green={green_time_ns}s, Yellow={yellow_time}s, All-red={all_red_time}s")
                print(f"    E-W: Green={green_time_ew}s, Yellow={yellow_time}s, All-red={all_red_time}s")
        
        # Override the method
        simulation._optimize_traffic_signals = fixed_timing.__get__(simulation)
    
    # Set simulation duration
    simulation.max_simulation_time = args.duration
    
    # Run simulation
    start_time = time.time()
    simulation.run_simulation()
    end_time = time.time()
    
    # Generate performance report
    report_dir = simulation.generate_performance_report()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Performance report saved to {report_dir}")

def run_tests(args):
    """Run model tests and evaluation."""
    print("Starting AI-Driven Smart Traffic Congestion Model Testing...")
    print(f"Output directory: {args.output}")
    
    # Create tester
    tester = ModelTester(output_dir=args.output)
    
    # Run tests
    start_time = time.time()
    tester.run_tests()
    end_time = time.time()
    
    print(f"Testing completed in {end_time - start_time:.2f} seconds")
    print(f"Test results saved to {args.output}")

def run_component(args):
    """Run an individual component."""
    if args.component is None:
        print("Error: Component must be specified in component mode")
        sys.exit(1)
    
    print(f"Starting component: {args.component}")
    print(f"Output directory: {args.output}")
    
    # Create output directory for component
    component_dir = os.path.join(args.output, args.component)
    os.makedirs(component_dir, exist_ok=True)
    
    if args.component == 'processor':
        # Run distributed image processor
        processor = DistributedImageProcessor()
        print("Distributed Image Processor initialized")
        print("This component requires images to process. Please use it within the simulation.")
    
    elif args.component == 'detector':
        # Run YOLOv5 vehicle detector
        detector = YOLOv5VehicleDetector()
        print("YOLOv5 Vehicle Detector initialized")
        print("This component requires images to process. Please use it within the simulation.")
    
    elif args.component == 'predictor':
        # Run RNN-LSTM predictor
        predictor = TrafficVolumePredictor()
        
        # Generate synthetic data for demonstration
        from scripts.rnn_lstm_predictor import generate_synthetic_traffic_data
        data = generate_synthetic_traffic_data(num_days=7)
        
        # Save data
        data_path = os.path.join(component_dir, "synthetic_traffic_data.csv")
        data.to_csv(data_path, index=False)
        print(f"Synthetic traffic data saved to {data_path}")
        
        # Train predictor
        print("Training RNN-LSTM predictor...")
        predictor.train(data, epochs=10, verbose=1)
        
        # Make predictions
        print("Making predictions...")
        predictions = predictor.predict(data.tail(24))
        
        # Save predictions
        pred_path = os.path.join(component_dir, "traffic_predictions.csv")
        predictions.to_csv(pred_path, index=False)
        print(f"Traffic predictions saved to {pred_path}")
        
        # Visualize predictions
        print("Generating visualization...")
        fig = predictor.visualize_predictions(data.tail(24), predictions)
        fig_path = os.path.join(component_dir, "prediction_visualization.png")
        fig.savefig(fig_path)
        print(f"Prediction visualization saved to {fig_path}")
    
    elif args.component == 'controller':
        # Run fuzzy logic controller
        controller = FuzzyTrafficController()
        
        # Visualize membership functions
        print("Generating membership function visualizations...")
        figs_membership = controller.visualize_membership_functions(component_dir)
        print(f"Membership function visualizations saved to {component_dir}")
        
        # Visualize control surface
        print("Generating control surface visualizations...")
        figs_surface = controller.visualize_control_surface(component_dir)
        print(f"Control surface visualizations saved to {component_dir}")
        
        # Test with various traffic conditions
        print("\nTesting controller with various traffic conditions:")
        
        test_conditions = [
            {"name": "Light traffic", "current": 30, "predicted": 40, "queue": 5, "wait": 15},
            {"name": "Medium traffic", "current": 100, "predicted": 110, "queue": 20, "wait": 45},
            {"name": "Heavy traffic", "current": 180, "predicted": 190, "queue": 40, "wait": 90},
            {"name": "Increasing traffic", "current": 80, "predicted": 150, "queue": 15, "wait": 30},
            {"name": "Decreasing traffic", "current": 120, "predicted": 60, "queue": 25, "wait": 60}
        ]
        
        import pandas as pd
        results = []
        
        for condition in test_conditions:
            print(f"\nCondition: {condition['name']}")
            print(f"  Current volume: {condition['current']}")
            print(f"  Predicted volume: {condition['predicted']}")
            print(f"  Queue length: {condition['queue']}")
            print(f"  Waiting time: {condition['wait']}")
            
            # Compute signal timing
            timing = controller.compute_signal_timing(
                condition['current'], condition['predicted'], 
                condition['queue'], condition['wait']
            )
            
            print(f"  Results:")
            print(f"    Cycle length: {timing['cycle_length']:.2f} seconds")
            print(f"    Green proportion (N-S): {timing['green_proportion']:.2f}")
            print(f"    Green time (N-S): {timing['green_time_ns']:.2f} seconds")
            print(f"    Green time (E-W): {timing['green_time_ew']:.2f} seconds")
            
            # Generate timing plan
            plan = controller.generate_timing_plan(f"test_{condition['name'].lower().replace(' ', '_')}", timing)
            
            # Save results
            results.append({
                "condition": condition['name'],
                "current_volume": condition['current'],
                "predicted_volume": condition['predicted'],
                "queue_length": condition['queue'],
                "waiting_time": condition['wait'],
                "cycle_length": timing['cycle_length'],
                "green_proportion": timing['green_proportion'],
                "green_time_ns": timing['green_time_ns'],
                "green_time_ew": timing['green_time_ew']
            })
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_path = os.path.join(component_dir, "fuzzy_controller_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
    
    else:
        print(f"Unknown component: {args.component}")
        sys.exit(1)
    
    print(f"Component {args.component} completed")

if __name__ == "__main__":
    main()
