#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fuzzy Logic Controller for Adaptive Traffic Signal Control
This module implements a fuzzy logic controller for adaptive traffic signal control
in the AI-driven smart traffic congestion model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyTrafficController:
    """
    A class for adaptive traffic signal control using fuzzy logic.
    This class provides functionality to control traffic signals based on
    current traffic conditions and predicted volumes.
    """
    
    def __init__(self):
        """Initialize the fuzzy traffic controller."""
        # Create fuzzy control system
        self.system = self._create_fuzzy_system()
        self.simulation = ctrl.ControlSystemSimulation(self.system)
        
        # Default cycle lengths (in seconds)
        self.min_cycle_length = 30
        self.max_cycle_length = 180
        
        # Default green time proportions
        self.min_green_proportion = 0.3
        self.max_green_proportion = 0.7
        
        # Default phase sequence
        self.phase_sequence = ['N-S', 'E-W']
    
    def _create_fuzzy_system(self):
        """
        Create the fuzzy control system.
        
        Returns:
            ControlSystem: Fuzzy control system
        """
        # Define input variables
        current_volume = ctrl.Antecedent(np.arange(0, 201, 1), 'current_volume')
        predicted_volume = ctrl.Antecedent(np.arange(0, 201, 1), 'predicted_volume')
        queue_length = ctrl.Antecedent(np.arange(0, 51, 1), 'queue_length')
        waiting_time = ctrl.Antecedent(np.arange(0, 121, 1), 'waiting_time')
        
        # Define output variables
        cycle_length = ctrl.Consequent(np.arange(30, 181, 1), 'cycle_length')
        green_proportion = ctrl.Consequent(np.arange(0.3, 0.71, 0.01), 'green_proportion')
        
        # Define membership functions for inputs
        # Current volume
        current_volume['low'] = fuzz.trimf(current_volume.universe, [0, 0, 70])
        current_volume['medium'] = fuzz.trimf(current_volume.universe, [50, 100, 150])
        current_volume['high'] = fuzz.trimf(current_volume.universe, [130, 200, 200])
        
        # Predicted volume
        predicted_volume['decreasing'] = fuzz.trimf(predicted_volume.universe, [0, 0, 80])
        predicted_volume['steady'] = fuzz.trimf(predicted_volume.universe, [60, 100, 140])
        predicted_volume['increasing'] = fuzz.trimf(predicted_volume.universe, [120, 200, 200])
        
        # Queue length
        queue_length['short'] = fuzz.trimf(queue_length.universe, [0, 0, 15])
        queue_length['medium'] = fuzz.trimf(queue_length.universe, [10, 25, 40])
        queue_length['long'] = fuzz.trimf(queue_length.universe, [35, 50, 50])
        
        # Waiting time
        waiting_time['short'] = fuzz.trimf(waiting_time.universe, [0, 0, 30])
        waiting_time['medium'] = fuzz.trimf(waiting_time.universe, [20, 50, 80])
        waiting_time['long'] = fuzz.trimf(waiting_time.universe, [70, 120, 120])
        
        # Define membership functions for outputs
        # Cycle length
        cycle_length['short'] = fuzz.trimf(cycle_length.universe, [30, 30, 80])
        cycle_length['medium'] = fuzz.trimf(cycle_length.universe, [60, 105, 150])
        cycle_length['long'] = fuzz.trimf(cycle_length.universe, [130, 180, 180])
        
        # Green proportion
        green_proportion['low'] = fuzz.trimf(green_proportion.universe, [0.3, 0.3, 0.45])
        green_proportion['medium'] = fuzz.trimf(green_proportion.universe, [0.4, 0.5, 0.6])
        green_proportion['high'] = fuzz.trimf(green_proportion.universe, [0.55, 0.7, 0.7])
        
        # Define fuzzy rules
        rules = [
            # Rules for cycle length
            ctrl.Rule(current_volume['low'] & predicted_volume['decreasing'], cycle_length['short']),
            ctrl.Rule(current_volume['low'] & predicted_volume['steady'], cycle_length['short']),
            ctrl.Rule(current_volume['low'] & predicted_volume['increasing'], cycle_length['medium']),
            ctrl.Rule(current_volume['medium'] & predicted_volume['decreasing'], cycle_length['short']),
            ctrl.Rule(current_volume['medium'] & predicted_volume['steady'], cycle_length['medium']),
            ctrl.Rule(current_volume['medium'] & predicted_volume['increasing'], cycle_length['long']),
            ctrl.Rule(current_volume['high'] & predicted_volume['decreasing'], cycle_length['medium']),
            ctrl.Rule(current_volume['high'] & predicted_volume['steady'], cycle_length['long']),
            ctrl.Rule(current_volume['high'] & predicted_volume['increasing'], cycle_length['long']),
            
            # Rules for green proportion (N-S vs E-W)
            ctrl.Rule(queue_length['short'] & waiting_time['short'], green_proportion['medium']),
            ctrl.Rule(queue_length['short'] & waiting_time['medium'], green_proportion['medium']),
            ctrl.Rule(queue_length['short'] & waiting_time['long'], green_proportion['high']),
            ctrl.Rule(queue_length['medium'] & waiting_time['short'], green_proportion['medium']),
            ctrl.Rule(queue_length['medium'] & waiting_time['medium'], green_proportion['medium']),
            ctrl.Rule(queue_length['medium'] & waiting_time['long'], green_proportion['high']),
            ctrl.Rule(queue_length['long'] & waiting_time['short'], green_proportion['high']),
            ctrl.Rule(queue_length['long'] & waiting_time['medium'], green_proportion['high']),
            ctrl.Rule(queue_length['long'] & waiting_time['long'], green_proportion['high']),
        ]
        
        # Create control system
        system = ctrl.ControlSystem(rules)
        
        return system
    
    def compute_signal_timing(self, current_volume, predicted_volume, queue_length, waiting_time):
        """
        Compute signal timing based on current traffic conditions and predictions.
        
        Args:
            current_volume (float): Current traffic volume
            predicted_volume (float): Predicted traffic volume
            queue_length (float): Current queue length
            waiting_time (float): Current waiting time
            
        Returns:
            dict: Dictionary with signal timing parameters
        """
        # Set input values
        self.simulation.input['current_volume'] = current_volume
        self.simulation.input['predicted_volume'] = predicted_volume
        self.simulation.input['queue_length'] = queue_length
        self.simulation.input['waiting_time'] = waiting_time
        
        # Compute output
        self.simulation.compute()
        
        # Get output values
        cycle_length = self.simulation.output['cycle_length']
        green_proportion = self.simulation.output['green_proportion']
        
        # Calculate green times for each phase
        green_time_ns = cycle_length * green_proportion
        green_time_ew = cycle_length * (1 - green_proportion)
        
        return {
            'cycle_length': cycle_length,
            'green_proportion': green_proportion,
            'green_time_ns': green_time_ns,
            'green_time_ew': green_time_ew
        }
    
    def optimize_signal_timing(self, traffic_data, prediction_data):
        """
        Optimize signal timing for multiple intersections based on traffic data and predictions.
        
        Args:
            traffic_data (dict): Dictionary with current traffic data for each intersection
            prediction_data (dict): Dictionary with predicted traffic data for each intersection
            
        Returns:
            dict: Dictionary with optimized signal timing for each intersection
        """
        optimized_timing = {}
        
        for intersection_id, data in traffic_data.items():
            # Get current traffic conditions
            current_volume = data.get('volume', 0)
            queue_length = data.get('queue_length', 0)
            waiting_time = data.get('waiting_time', 0)
            
            # Get predicted volume
            predicted_volume = 0
            if intersection_id in prediction_data:
                predicted_volume = prediction_data[intersection_id].get('predicted_volume', 0)
            
            # Compute signal timing
            timing = self.compute_signal_timing(
                current_volume, predicted_volume, queue_length, waiting_time)
            
            optimized_timing[intersection_id] = timing
        
        return optimized_timing
    
    def visualize_membership_functions(self, output_dir=None):
        """
        Visualize membership functions.
        
        Args:
            output_dir (str): Directory to save visualization results
        """
        # Create figures
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
        fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Create simplified visualizations of membership functions
        # Current volume
        x_current = np.arange(0, 201, 1)
        low = fuzz.trimf(x_current, [0, 0, 70])
        medium = fuzz.trimf(x_current, [50, 100, 150])
        high = fuzz.trimf(x_current, [130, 200, 200])
        
        ax1.plot(x_current, low, 'b', linewidth=1.5, label='Low')
        ax1.plot(x_current, medium, 'g', linewidth=1.5, label='Medium')
        ax1.plot(x_current, high, 'r', linewidth=1.5, label='High')
        ax1.set_title('Current Volume')
        ax1.legend()
        
        # Predicted volume
        x_predicted = np.arange(0, 201, 1)
        decreasing = fuzz.trimf(x_predicted, [0, 0, 80])
        steady = fuzz.trimf(x_predicted, [60, 100, 140])
        increasing = fuzz.trimf(x_predicted, [120, 200, 200])
        
        ax2.plot(x_predicted, decreasing, 'b', linewidth=1.5, label='Decreasing')
        ax2.plot(x_predicted, steady, 'g', linewidth=1.5, label='Steady')
        ax2.plot(x_predicted, increasing, 'r', linewidth=1.5, label='Increasing')
        ax2.set_title('Predicted Volume')
        ax2.legend()
        
        # Queue length
        x_queue = np.arange(0, 51, 1)
        short = fuzz.trimf(x_queue, [0, 0, 15])
        medium_q = fuzz.trimf(x_queue, [10, 25, 40])
        long = fuzz.trimf(x_queue, [35, 50, 50])
        
        ax3.plot(x_queue, short, 'b', linewidth=1.5, label='Short')
        ax3.plot(x_queue, medium_q, 'g', linewidth=1.5, label='Medium')
        ax3.plot(x_queue, long, 'r', linewidth=1.5, label='Long')
        ax3.set_title('Queue Length')
        ax3.legend()
        
        # Waiting time
        x_wait = np.arange(0, 121, 1)
        short_w = fuzz.trimf(x_wait, [0, 0, 30])
        medium_w = fuzz.trimf(x_wait, [20, 50, 80])
        long_w = fuzz.trimf(x_wait, [70, 120, 120])
        
        ax4.plot(x_wait, short_w, 'b', linewidth=1.5, label='Short')
        ax4.plot(x_wait, medium_w, 'g', linewidth=1.5, label='Medium')
        ax4.plot(x_wait, long_w, 'r', linewidth=1.5, label='Long')
        ax4.set_title('Waiting Time')
        ax4.legend()
        
        # Cycle length
        x_cycle = np.arange(30, 181, 1)
        short_c = fuzz.trimf(x_cycle, [30, 30, 80])
        medium_c = fuzz.trimf(x_cycle, [60, 105, 150])
        long_c = fuzz.trimf(x_cycle, [130, 180, 180])
        
        ax5.plot(x_cycle, short_c, 'b', linewidth=1.5, label='Short')
        ax5.plot(x_cycle, medium_c, 'g', linewidth=1.5, label='Medium')
        ax5.plot(x_cycle, long_c, 'r', linewidth=1.5, label='Long')
        ax5.set_title('Cycle Length')
        ax5.legend()
        
        # Green proportion
        x_green = np.arange(0.3, 0.71, 0.01)
        low_g = fuzz.trimf(x_green, [0.3, 0.3, 0.45])
        medium_g = fuzz.trimf(x_green, [0.4, 0.5, 0.6])
        high_g = fuzz.trimf(x_green, [0.55, 0.7, 0.7])
        
        ax6.plot(x_green, low_g, 'b', linewidth=1.5, label='Low')
        ax6.plot(x_green, medium_g, 'g', linewidth=1.5, label='Medium')
        ax6.plot(x_green, high_g, 'r', linewidth=1.5, label='High')
        ax6.set_title('Green Proportion (N-S)')
        ax6.legend()
        
        # Adjust layout
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        
        # Save figures if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig1.savefig(os.path.join(output_dir, 'volume_membership.png'))
            fig2.savefig(os.path.join(output_dir, 'queue_wait_membership.png'))
            fig3.savefig(os.path.join(output_dir, 'output_membership.png'))
        
        return [fig1, fig2, fig3]
    
    def visualize_control_surface(self, output_dir=None):
        """
        Visualize control surface.
        
        Args:
            output_dir (str): Directory to save visualization results
        """
        # Create figures
        fig1 = plt.figure(figsize=(10, 8))
        fig2 = plt.figure(figsize=(10, 8))
        
        # Plot control surfaces
        ax1 = fig1.add_subplot(111, projection='3d')
        ax2 = fig2.add_subplot(111, projection='3d')
        
        # For simplified testing, create basic surface plots
        # instead of using the actual control system view
        
        # Create sample data for visualization
        x = np.linspace(0, 200, 30)  # current_volume / queue_length
        y = np.linspace(0, 200, 30)  # predicted_volume / waiting_time
        X, Y = np.meshgrid(x, y)
        
        # Simplified cycle length function (just for visualization)
        Z1 = 30 + 150 * (0.3 * X/200 + 0.7 * Y/200)
        
        # Simplified green proportion function (just for visualization)
        Z2 = 0.3 + 0.4 * (0.6 * X/50 + 0.4 * Y/120)
        Z2 = np.clip(Z2, 0.3, 0.7)
        
        # Plot surfaces
        surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Current Volume')
        ax1.set_ylabel('Predicted Volume')
        ax1.set_zlabel('Cycle Length (s)')
        ax1.set_title('Cycle Length Control Surface (Simplified)')
        
        surf2 = ax2.plot_surface(X, Y, Z2, cmap='viridis', alpha=0.8)
        ax2.set_xlabel('Queue Length')
        ax2.set_ylabel('Waiting Time')
        ax2.set_zlabel('Green Proportion (N-S)')
        ax2.set_title('Green Proportion Control Surface (Simplified)')
        
        # Adjust layout
        fig1.tight_layout()
        fig2.tight_layout()
        
        # Save figures if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig1.savefig(os.path.join(output_dir, 'cycle_length_surface.png'))
            fig2.savefig(os.path.join(output_dir, 'green_proportion_surface.png'))
        
        return [fig1, fig2]
    
    def generate_timing_plan(self, intersection_id, timing):
        """
        Generate a timing plan for an intersection.
        
        Args:
            intersection_id (str): Intersection ID
            timing (dict): Signal timing parameters
            
        Returns:
            dict: Timing plan
        """
        cycle_length = timing['cycle_length']
        green_time_ns = timing['green_time_ns']
        green_time_ew = timing['green_time_ew']
        
        # Calculate yellow and all-red times
        yellow_time = 3  # seconds
        all_red_time = 2  # seconds
        
        # Calculate effective green times
        effective_green_ns = green_time_ns - yellow_time - all_red_time
        effective_green_ew = green_time_ew - yellow_time - all_red_time
        
        # Create timing plan
        timing_plan = {
            'intersection_id': intersection_id,
            'cycle_length': cycle_length,
            'phases': [
                {
                    'id': 1,
                    'direction': 'N-S',
                    'green_time': effective_green_ns,
                    'yellow_time': yellow_time,
                    'all_red_time': all_red_time
                },
                {
                    'id': 2,
                    'direction': 'E-W',
                    'green_time': effective_green_ew,
                    'yellow_time': yellow_time,
                    'all_red_time': all_red_time
                }
            ]
        }
        
        return timing_plan


# Example usage
if __name__ == "__main__":
    # Initialize controller
    controller = FuzzyTrafficController()
    
    # Visualize membership functions
    controller.visualize_membership_functions('fuzzy_membership')
    
    # Visualize control surface
    controller.visualize_control_surface('fuzzy_surface')
    
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
        }
    }
    
    # Example prediction data
    prediction_data = {
        'intersection_1': {
            'predicted_volume': 150
        },
        'intersection_2': {
            'predicted_volume': 90
        }
    }
    
    # Optimize signal timing
    optimized_timing = controller.optimize_signal_timing(traffic_data, prediction_data)
    
    # Generate timing plans
    timing_plans = {}
    for intersection_id, timing in optimized_timing.items():
        timing_plans[intersection_id] = controller.generate_timing_plan(intersection_id, timing)
    
    # Print timing plans
    for intersection_id, plan in timing_plans.items():
        print(f"Timing plan for {intersection_id}:")
        print(f"Cycle length: {plan['cycle_length']:.2f} seconds")
        for phase in plan['phases']:
            print(f"  Phase {phase['id']} ({phase['direction']}): "
                 f"Green = {phase['green_time']:.2f}s, "
                 f"Yellow = {phase['yellow_time']}s, "
                 f"All-red = {phase['all_red_time']}s")
        print()
