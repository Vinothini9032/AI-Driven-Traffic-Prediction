#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Fuzzy Logic Traffic Controller
This script tests the functionality of the FuzzyTrafficController class.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzy_logic_controller import FuzzyTrafficController

def test_fuzzy_traffic_controller():
    """Test the Fuzzy Logic Traffic Controller functionality."""
    print("Testing Fuzzy Logic Traffic Controller...")
    
    # Create output directory for results
    output_dir = "/home/ubuntu/traffic_congestion_model/output/fuzzy_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize controller
    print("Initializing Fuzzy Logic Controller...")
    controller = FuzzyTrafficController()
    
    # Visualize membership functions
    print("Visualizing membership functions...")
    figs_membership = controller.visualize_membership_functions(output_dir)
    print(f"Membership function visualizations saved to {output_dir}")
    
    # Visualize control surface
    print("Visualizing control surfaces...")
    figs_surface = controller.visualize_control_surface(output_dir)
    print(f"Control surface visualizations saved to {output_dir}")
    
    # Close figures to free memory
    for fig in figs_membership + figs_surface:
        plt.close(fig)
    
    # Test with various traffic conditions
    print("\nTesting controller with various traffic conditions:")
    
    test_conditions = [
        {"name": "Light traffic", "current": 30, "predicted": 40, "queue": 5, "wait": 15},
        {"name": "Medium traffic", "current": 100, "predicted": 110, "queue": 20, "wait": 45},
        {"name": "Heavy traffic", "current": 180, "predicted": 190, "queue": 40, "wait": 90},
        {"name": "Increasing traffic", "current": 80, "predicted": 150, "queue": 15, "wait": 30},
        {"name": "Decreasing traffic", "current": 120, "predicted": 60, "queue": 25, "wait": 60}
    ]
    
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
    results_df.to_csv(f"{output_dir}/fuzzy_controller_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/fuzzy_controller_results.csv")
    
    # Test with multiple intersections
    print("\nTesting controller with multiple intersections:")
    
    # Example traffic data
    traffic_data = {
        'intersection_1': {
            'volume': 120,
            'queue_length': 15,
            'waiting_time': 45
        },
        'intersection_2': {
            'volume': 80,
            'queue_length': 10,
            'waiting_time': 30
        },
        'intersection_3': {
            'volume': 160,
            'queue_length': 35,
            'waiting_time': 75
        }
    }
    
    # Example prediction data
    prediction_data = {
        'intersection_1': {
            'predicted_volume': 150
        },
        'intersection_2': {
            'predicted_volume': 90
        },
        'intersection_3': {
            'predicted_volume': 140
        }
    }
    
    # Optimize signal timing
    print("Optimizing signal timing for multiple intersections...")
    optimized_timing = controller.optimize_signal_timing(traffic_data, prediction_data)
    
    # Generate timing plans
    timing_plans = {}
    for intersection_id, timing in optimized_timing.items():
        timing_plans[intersection_id] = controller.generate_timing_plan(intersection_id, timing)
    
    # Print and save timing plans
    intersection_results = []
    
    for intersection_id, plan in timing_plans.items():
        print(f"\nTiming plan for {intersection_id}:")
        print(f"Cycle length: {plan['cycle_length']:.2f} seconds")
        
        for phase in plan['phases']:
            print(f"  Phase {phase['id']} ({phase['direction']}): "
                 f"Green = {phase['green_time']:.2f}s, "
                 f"Yellow = {phase['yellow_time']}s, "
                 f"All-red = {phase['all_red_time']}s")
        
        # Save results
        intersection_results.append({
            "intersection_id": intersection_id,
            "current_volume": traffic_data[intersection_id]['volume'],
            "predicted_volume": prediction_data[intersection_id]['predicted_volume'],
            "queue_length": traffic_data[intersection_id]['queue_length'],
            "waiting_time": traffic_data[intersection_id]['waiting_time'],
            "cycle_length": plan['cycle_length'],
            "green_time_ns": plan['phases'][0]['green_time'],
            "yellow_time_ns": plan['phases'][0]['yellow_time'],
            "green_time_ew": plan['phases'][1]['green_time'],
            "yellow_time_ew": plan['phases'][1]['yellow_time']
        })
    
    # Save intersection results to CSV
    intersection_df = pd.DataFrame(intersection_results)
    intersection_df.to_csv(f"{output_dir}/intersection_timing_plans.csv", index=False)
    print(f"\nIntersection timing plans saved to {output_dir}/intersection_timing_plans.csv")
    
    # Create a visualization of the timing plans
    print("Creating visualization of timing plans...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up bar positions
    intersections = list(timing_plans.keys())
    x = np.arange(len(intersections))
    width = 0.35
    
    # Extract data for plotting
    cycle_lengths = [timing_plans[i]['cycle_length'] for i in intersections]
    green_times_ns = [timing_plans[i]['phases'][0]['green_time'] for i in intersections]
    green_times_ew = [timing_plans[i]['phases'][1]['green_time'] for i in intersections]
    
    # Create bars
    ax.bar(x - width/2, green_times_ns, width, label='N-S Green Time')
    ax.bar(x + width/2, green_times_ew, width, label='E-W Green Time')
    
    # Add cycle length as text
    for i, v in enumerate(cycle_lengths):
        ax.text(i, max(green_times_ns[i], green_times_ew[i]) + 5, 
                f"Cycle: {v:.1f}s", ha='center')
    
    # Add labels and legend
    ax.set_xlabel('Intersection')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Signal Timing Plans by Intersection')
    ax.set_xticks(x)
    ax.set_xticklabels(intersections)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/timing_plans_comparison.png")
    plt.close(fig)
    print(f"Timing plans visualization saved to {output_dir}/timing_plans_comparison.png")
    
    print("\nTest completed successfully!")
    return True

if __name__ == "__main__":
    test_fuzzy_traffic_controller()
