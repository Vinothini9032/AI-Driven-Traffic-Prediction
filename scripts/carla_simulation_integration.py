#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA Simulation Integration for AI-Driven Smart Traffic Congestion Model
This module integrates the distributed processing, YOLOv5 detection, 
RNN-LSTM prediction, and fuzzy logic controller with CARLA simulation.
"""

import os
import sys
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pygame


from .distributed_image_processor import DistributedImageProcessor
from .yolov5_vehicle_detector import YOLOv5VehicleDetector
from .rnn_lstm_predictor import TrafficVolumePredictor, generate_synthetic_traffic_data
from .fuzzy_logic_controller import FuzzyTrafficController


# Import our custom modules


class CARLASimulationIntegration:
    """
    A class for integrating AI components with CARLA simulation.
    This class provides functionality to run a simulated traffic environment
    and apply the AI-driven traffic congestion model.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the CARLA simulation integration.
        
        Args:
            output_dir (str): Directory to save output files
        """
        self.output_dir = output_dir or "/home/ubuntu/traffic_congestion_model/output/carla_simulation"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.image_processor = None
        self.vehicle_detector = None
        self.traffic_predictor = None
        self.traffic_controller = None
        
        # Simulation parameters
        self.simulation_time = 0
        self.time_step = 1  # seconds
        self.max_simulation_time = 3600  # 1 hour
        self.intersections = {}
        self.vehicles = {}
        self.traffic_lights = {}
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        
        # Pygame for visualization
        self.screen = None
        self.clock = None
        self.font = None
        
        # Chennai map parameters (simplified for simulation)
        self.map_width = 1000
        self.map_height = 800
        self.intersections_layout = [
            {"id": "int_1", "x": 300, "y": 300, "roads": ["NS", "EW"]},
            {"id": "int_2", "x": 700, "y": 300, "roads": ["NS", "EW"]},
            {"id": "int_3", "x": 300, "y": 600, "roads": ["NS", "EW"]},
            {"id": "int_4", "x": 700, "y": 600, "roads": ["NS", "EW"]}
        ]
    
    def initialize_components(self):
        """Initialize all AI components."""
        print("Initializing AI components...")
        
        # Initialize distributed image processor
        self.image_processor = DistributedImageProcessor()
        
        # Initialize YOLOv5 vehicle detector
        self.vehicle_detector = YOLOv5VehicleDetector()
        
        # Initialize RNN-LSTM traffic predictor
        self.traffic_predictor = TrafficVolumePredictor(sequence_length=12, prediction_horizon=12)
        
        # Initialize fuzzy logic traffic controller
        self.traffic_controller = FuzzyTrafficController()
        
        print("All components initialized successfully.")
    
    def setup_simulation(self):
        """Set up the simulation environment."""
        print("Setting up simulation environment...")
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.map_width, self.map_height))
        pygame.display.set_caption("AI-Driven Smart Traffic Congestion Model - Chennai Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        
        # Create intersections
        for intersection_data in self.intersections_layout:
            intersection_id = intersection_data["id"]
            x, y = intersection_data["x"], intersection_data["y"]
            roads = intersection_data["roads"]
            
            # Create intersection
            self.intersections[intersection_id] = {
                "id": intersection_id,
                "position": (x, y),
                "roads": roads,
                "traffic_lights": {},
                "vehicles": {},
                "queue_length": {"NS": 0, "EW": 0},
                "waiting_time": {"NS": 0, "EW": 0},
                "volume": {"NS": 0, "EW": 0},
                "predicted_volume": {"NS": 0, "EW": 0}
            }
            
            # Create traffic lights for each road
            for road in roads:
                self.traffic_lights[f"{intersection_id}_{road}"] = {
                    "id": f"{intersection_id}_{road}",
                    "intersection_id": intersection_id,
                    "road": road,
                    "state": "red",
                    "timer": 0,
                    "cycle_length": 60,
                    "green_time": 25,
                    "yellow_time": 5,
                    "all_red_time": 2
                }
        
        # Generate initial traffic data for prediction
        self._generate_initial_traffic_data()
        
        print("Simulation environment set up successfully.")
    
    def _generate_initial_traffic_data(self):
        """Generate initial traffic data for prediction."""
        # Generate synthetic data for each intersection
        for intersection_id, intersection in self.intersections.items():
            # Generate data for past 7 days (hourly data)
            data = generate_synthetic_traffic_data(num_days=7)
            
            # Save data
            data_path = os.path.join(self.output_dir, f"{intersection_id}_traffic_data.csv")
            data.to_csv(data_path, index=False)
            
            # Train predictor on this data
            self.traffic_predictor.train(data, epochs=5, verbose=0)
            
            # Make initial predictions
            predictions = self.traffic_predictor.predict(data.tail(24))
            
            # Set predicted volume (average of next hour)
            next_hour_predictions = predictions.head(12)
            avg_volume = next_hour_predictions['predicted_volume'].mean()
            intersection["predicted_volume"]["NS"] = avg_volume * 0.6  # 60% for NS
            intersection["predicted_volume"]["EW"] = avg_volume * 0.4  # 40% for EW
    
    def run_simulation(self):
        """Run the simulation."""
        print("Starting simulation...")
        self.is_running = True
        
        # Main simulation loop
        while self.is_running and self.simulation_time < self.max_simulation_time:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False
                    elif event.key == pygame.K_SPACE:
                        self.is_paused = not self.is_paused
            
            if not self.is_paused:
                # Update simulation
                self._update_simulation()
                
                # Render simulation
                self._render_simulation()
                
                # Increment simulation time
                self.simulation_time += self.time_step
            
            # Cap the frame rate
            self.clock.tick(10)
        
        # Clean up
        pygame.quit()
        print(f"Simulation completed. Total simulation time: {self.simulation_time} seconds.")
    
    def _update_simulation(self):
        """Update the simulation state."""
        # Update traffic volumes based on time of day
        self._update_traffic_volumes()
        
        # Update vehicle positions
        self._update_vehicles()
        
        # Update traffic lights
        self._update_traffic_lights()
        
        # Every 5 minutes in simulation time
        if self.simulation_time % 300 == 0:
            # Process traffic images (simulated)
            self._process_traffic_images()
            
            # Update traffic predictions
            self._update_traffic_predictions()
            
            # Optimize traffic signal timing
            self._optimize_traffic_signals()
    
    def _update_traffic_volumes(self):
        """Update traffic volumes based on time of day."""
        # Simulate time of day effect on traffic volume
        # Convert simulation time to time of day (assuming simulation starts at 8:00 AM)
        start_hour = 8
        current_hour = (start_hour + self.simulation_time // 3600) % 24
        current_minute = (self.simulation_time % 3600) // 60
        
        # Traffic volume factors by hour (simplified)
        hour_factors = {
            0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.1, 5: 0.2,
            6: 0.4, 7: 0.7, 8: 1.0, 9: 0.9, 10: 0.8, 11: 0.7,
            12: 0.8, 13: 0.8, 14: 0.7, 15: 0.8, 16: 0.9, 17: 1.0,
            18: 0.9, 19: 0.7, 20: 0.5, 21: 0.3, 22: 0.2, 23: 0.1
        }
        
        # Get current factor
        current_factor = hour_factors[current_hour]
        
        # Update volumes for each intersection
        for intersection_id, intersection in self.intersections.items():
            # Base volume (vehicles per minute)
            base_volume_ns = 15 + random.randint(-2, 2)  # 15 ± 2 vehicles/min for NS
            base_volume_ew = 12 + random.randint(-2, 2)  # 12 ± 2 vehicles/min for EW
            
            # Apply time factor
            intersection["volume"]["NS"] = base_volume_ns * current_factor
            intersection["volume"]["EW"] = base_volume_ew * current_factor
            
            # Add some randomness
            intersection["volume"]["NS"] *= random.uniform(0.9, 1.1)
            intersection["volume"]["EW"] *= random.uniform(0.9, 1.1)
    
    def _update_vehicles(self):
        """Update vehicle positions and generate new vehicles."""
        # For each intersection
        for intersection_id, intersection in self.intersections.items():
            # Get position
            x, y = intersection["position"]
            
            # For each road
            for road in intersection["roads"]:
                # Get traffic light
                traffic_light = self.traffic_lights[f"{intersection_id}_{road}"]
                
                # Generate new vehicles based on volume
                volume = intersection["volume"][road]
                
                # Probability of new vehicle per time step
                prob_new_vehicle = volume / 60.0  # Convert from vehicles/min to vehicles/sec
                
                if random.random() < prob_new_vehicle:
                    # Generate new vehicle
                    vehicle_id = f"veh_{intersection_id}_{road}_{self.simulation_time}"
                    
                    # Determine position based on road
                    if road == "NS":
                        # North-South road
                        veh_x = x
                        veh_y = y - 100 if random.random() < 0.5 else y + 100
                        direction = "S" if veh_y < y else "N"
                    else:
                        # East-West road
                        veh_x = x - 100 if random.random() < 0.5 else x + 100
                        veh_y = y
                        direction = "E" if veh_x < x else "W"
                    
                    # Create vehicle
                    self.vehicles[vehicle_id] = {
                        "id": vehicle_id,
                        "position": (veh_x, veh_y),
                        "intersection_id": intersection_id,
                        "road": road,
                        "direction": direction,
                        "speed": 0,
                        "waiting_time": 0,
                        "color": (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
                    }
                    
                    # Add to intersection
                    if vehicle_id not in intersection["vehicles"]:
                        intersection["vehicles"][vehicle_id] = self.vehicles[vehicle_id]
            
            # Update queue lengths and waiting times
            queue_ns = 0
            queue_ew = 0
            wait_ns = 0
            wait_ew = 0
            
            # Count vehicles in queue for each road
            for vehicle_id, vehicle in list(intersection["vehicles"].items()):
                # Remove vehicles that have passed through
                if self._is_vehicle_through_intersection(vehicle):
                    del intersection["vehicles"][vehicle_id]
                    if vehicle_id in self.vehicles:
                        del self.vehicles[vehicle_id]
                    continue
                
                # Update vehicle position based on traffic light
                road = vehicle["road"]
                traffic_light = self.traffic_lights[f"{intersection_id}_{road}"]
                
                # Check if vehicle should move
                if traffic_light["state"] == "green":
                    # Move vehicle
                    vehicle["speed"] = 5  # units per second
                    vehicle["waiting_time"] = 0
                else:
                    # Stop vehicle
                    vehicle["speed"] = 0
                    vehicle["waiting_time"] += self.time_step
                
                # Update position
                x, y = vehicle["position"]
                direction = vehicle["direction"]
                speed = vehicle["speed"]
                
                if direction == "N":
                    y -= speed
                elif direction == "S":
                    y += speed
                elif direction == "E":
                    x += speed
                elif direction == "W":
                    x -= speed
                
                vehicle["position"] = (x, y)
                
                # Count for queue and waiting time
                if vehicle["speed"] == 0:
                    if road == "NS":
                        queue_ns += 1
                        wait_ns += vehicle["waiting_time"]
                    else:
                        queue_ew += 1
                        wait_ew += vehicle["waiting_time"]
            
            # Update intersection queue lengths
            intersection["queue_length"]["NS"] = queue_ns
            intersection["queue_length"]["EW"] = queue_ew
            
            # Update intersection waiting times (average)
            intersection["waiting_time"]["NS"] = wait_ns / max(1, queue_ns)
            intersection["waiting_time"]["EW"] = wait_ew / max(1, queue_ew)
    
    def _is_vehicle_through_intersection(self, vehicle):
        """Check if a vehicle has passed through the intersection."""
        intersection_id = vehicle["intersection_id"]
        intersection_pos = self.intersections[intersection_id]["position"]
        int_x, int_y = intersection_pos
        veh_x, veh_y = vehicle["position"]
        direction = vehicle["direction"]
        
        # Check if vehicle has passed through based on direction
        if direction == "N" and veh_y < int_y - 50:
            return True
        elif direction == "S" and veh_y > int_y + 50:
            return True
        elif direction == "E" and veh_x > int_x + 50:
            return True
        elif direction == "W" and veh_x < int_x - 50:
            return True
        
        return False
    
    def _update_traffic_lights(self):
        """Update traffic light states."""
        for traffic_light_id, traffic_light in self.traffic_lights.items():
            # Update timer
            traffic_light["timer"] += self.time_step
            
            # Check if timer exceeds current phase duration
            if traffic_light["state"] == "green" and traffic_light["timer"] >= traffic_light["green_time"]:
                # Change to yellow
                traffic_light["state"] = "yellow"
                traffic_light["timer"] = 0
            elif traffic_light["state"] == "yellow" and traffic_light["timer"] >= traffic_light["yellow_time"]:
                # Change to red
                traffic_light["state"] = "red"
                traffic_light["timer"] = 0
            elif traffic_light["state"] == "red":
                # Check if opposite direction is also red and has been red for all-red time
                intersection_id = traffic_light["intersection_id"]
                road = traffic_light["road"]
                opposite_road = "EW" if road == "NS" else "NS"
                opposite_light = self.traffic_lights[f"{intersection_id}_{opposite_road}"]
                
                if opposite_light["state"] == "red" and opposite_light["timer"] >= opposite_light["all_red_time"]:
                    # Check if this light has been red for a full cycle minus green and yellow time
                    cycle_remaining = traffic_light["cycle_length"] - (
                        opposite_light["green_time"] + opposite_light["yellow_time"] + opposite_light["all_red_time"])
                    
                    if traffic_light["timer"] >= cycle_remaining:
                        # Change to green
                        traffic_light["state"] = "green"
                        traffic_light["timer"] = 0
    
    def _process_traffic_images(self):
        """Simulate processing of traffic camera images."""
        print(f"[{self._get_simulation_time_str()}] Processing traffic camera images...")
        
        # In a real implementation, this would capture images from CARLA
        # and process them using the distributed image processor and YOLOv5 detector
        
        # For simulation, we'll use the current vehicle counts as detection results
        for intersection_id, intersection in self.intersections.items():
            # Count vehicles by road
            ns_vehicles = sum(1 for v in intersection["vehicles"].values() if v["road"] == "NS")
            ew_vehicles = sum(1 for v in intersection["vehicles"].values() if v["road"] == "EW")
            
            # Log detection results
            print(f"  Intersection {intersection_id}: {ns_vehicles} N-S vehicles, {ew_vehicles} E-W vehicles detected")
    
    def _update_traffic_predictions(self):
        """Update traffic predictions using the RNN-LSTM model."""
        print(f"[{self._get_simulation_time_str()}] Updating traffic predictions...")
        
        # In a real implementation, this would use actual traffic data
        # For simulation, we'll use the current volumes with some projection
        
        for intersection_id, intersection in self.intersections.items():
            # Current volumes
            current_ns = intersection["volume"]["NS"]
            current_ew = intersection["volume"]["EW"]
            
            # Predict future volumes (simplified)
            # In reality, this would use the trained LSTM model
            time_of_day = (8 + self.simulation_time // 3600) % 24  # Assuming start at 8 AM
            
            # Simplified prediction based on time of day trends
            if 7 <= time_of_day < 10:  # Morning rush
                trend_factor = 1.2  # Increasing
            elif 16 <= time_of_day < 19:  # Evening rush
                trend_factor = 1.3  # Increasing
            elif 10 <= time_of_day < 16:  # Midday
                trend_factor = 0.9  # Slightly decreasing
            else:  # Night
                trend_factor = 0.7  # Decreasing
            
            # Apply trend with some randomness
            predicted_ns = current_ns * trend_factor * random.uniform(0.9, 1.1)
            predicted_ew = current_ew * trend_factor * random.uniform(0.9, 1.1)
            
            # Update intersection predictions
            intersection["predicted_volume"]["NS"] = predicted_ns
            intersection["predicted_volume"]["EW"] = predicted_ew
            
            print(f"  Intersection {intersection_id}: Predicted N-S volume: {predicted_ns:.2f}, E-W volume: {predicted_ew:.2f}")
    
    def _optimize_traffic_signals(self):
        """Optimize traffic signal timing using the fuzzy logic controller."""
        print(f"[{self._get_simulation_time_str()}] Optimizing traffic signal timing...")
        
        # Prepare traffic data for controller
        traffic_data = {}
        prediction_data = {}
        
        for intersection_id, intersection in self.intersections.items():
            # Average volume across both directions
            avg_volume = (intersection["volume"]["NS"] + intersection["volume"]["EW"]) / 2
            
            # Average queue length across both directions
            avg_queue = (intersection["queue_length"]["NS"] + intersection["queue_length"]["EW"]) / 2
            
            # Average waiting time across both directions
            avg_wait = (intersection["waiting_time"]["NS"] + intersection["waiting_time"]["EW"]) / 2
            
            # Prepare data
            traffic_data[intersection_id] = {
                "volume": avg_volume * 60,  # Convert to vehicles per hour
                "queue_length": avg_queue,
                "waiting_time": avg_wait
            }
            
            # Average predicted volume
            avg_predicted = (intersection["predicted_volume"]["NS"] + intersection["predicted_volume"]["EW"]) / 2
            
            prediction_data[intersection_id] = {
                "predicted_volume": avg_predicted * 60  # Convert to vehicles per hour
            }
        
        # Optimize signal timing
        optimized_timing = self.traffic_controller.optimize_signal_timing(traffic_data, prediction_data)
        
        # Generate timing plans
        timing_plans = {}
        for intersection_id, timing in optimized_timing.items():
            timing_plans[intersection_id] = self.traffic_controller.generate_timing_plan(intersection_id, timing)
        
        # Apply timing plans to traffic lights
        for intersection_id, plan in timing_plans.items():
            # Get cycle length
            cycle_length = plan["cycle_length"]
            
            # Apply to each phase
            for phase in plan["phases"]:
                direction = phase["direction"]
                green_time = phase["green_time"]
                yellow_time = phase["yellow_time"]
                all_red_time = phase["all_red_time"]
                
                # Map direction to road
                road = "NS" if direction == "N-S" else "EW"
                
                # Update traffic light
                traffic_light = self.traffic_lights[f"{intersection_id}_{road}"]
                traffic_light["cycle_length"] = cycle_length
                traffic_light["green_time"] = green_time
                traffic_light["yellow_time"] = yellow_time
                traffic_light["all_red_time"] = all_red_time
            
            print(f"  Intersection {intersection_id}: Cycle length set to {cycle_length:.2f}s")
            for phase in plan["phases"]:
                print(f"    {phase['direction']}: Green={phase['green_time']:.2f}s, Yellow={phase['yellow_time']}s, All-red={phase['all_red_time']}s")
    
    def _render_simulation(self):
        """Render the simulation."""
        # Clear screen
        self.screen.fill((240, 240, 240))
        
        # Draw roads
        self._draw_roads()
        
        # Draw intersections and traffic lights
        self._draw_intersections()
        
        # Draw vehicles
        self._draw_vehicles()
        
        # Draw HUD
        self._draw_hud()
        
        # Update display
        pygame.display.flip()
    
    def _draw_roads(self):
        """Draw roads on the screen."""
        for intersection_id, intersection in self.intersections.items():
            x, y = intersection["position"]
            
            # Draw roads
            for road in intersection["roads"]:
                if road == "NS":
                    # North-South road
                    pygame.draw.rect(self.screen, (100, 100, 100), (x - 20, 0, 40, self.map_height))
                    # Road markings
                    for i in range(0, self.map_height, 40):
                        pygame.draw.rect(self.screen, (255, 255, 255), (x - 2, i, 4, 20))
                else:
                    # East-West road
                    pygame.draw.rect(self.screen, (100, 100, 100), (0, y - 20, self.map_width, 40))
                    # Road markings
                    for i in range(0, self.map_width, 40):
                        pygame.draw.rect(self.screen, (255, 255, 255), (i, y - 2, 20, 4))
    
    def _draw_intersections(self):
        """Draw intersections and traffic lights on the screen."""
        for intersection_id, intersection in self.intersections.items():
            x, y = intersection["position"]
            
            # Draw intersection
            pygame.draw.rect(self.screen, (80, 80, 80), (x - 25, y - 25, 50, 50))
            
            # Draw traffic lights
            for road in intersection["roads"]:
                traffic_light = self.traffic_lights[f"{intersection_id}_{road}"]
                state = traffic_light["state"]
                
                # Determine color
                if state == "green":
                    color = (0, 255, 0)
                elif state == "yellow":
                    color = (255, 255, 0)
                else:
                    color = (255, 0, 0)
                
                # Draw traffic light
                if road == "NS":
                    # North traffic light
                    pygame.draw.circle(self.screen, color, (x, y - 30), 5)
                    # South traffic light
                    pygame.draw.circle(self.screen, color, (x, y + 30), 5)
                else:
                    # East traffic light
                    pygame.draw.circle(self.screen, color, (x + 30, y), 5)
                    # West traffic light
                    pygame.draw.circle(self.screen, color, (x - 30, y), 5)
    
    def _draw_vehicles(self):
        """Draw vehicles on the screen."""
        for vehicle_id, vehicle in self.vehicles.items():
            x, y = vehicle["position"]
            color = vehicle["color"]
            
            # Draw vehicle
            pygame.draw.rect(self.screen, color, (x - 5, y - 5, 10, 10))
    
    def _draw_hud(self):
        """Draw heads-up display with simulation information."""
        # Simulation time
        time_text = f"Simulation Time: {self._get_simulation_time_str()}"
        time_surface = self.font.render(time_text, True, (0, 0, 0))
        self.screen.blit(time_surface, (10, 10))
        
        # Simulation speed
        speed_text = f"Speed: {self.time_step}x"
        speed_surface = self.font.render(speed_text, True, (0, 0, 0))
        self.screen.blit(speed_surface, (10, 30))
        
        # Total vehicles
        vehicles_text = f"Vehicles: {len(self.vehicles)}"
        vehicles_surface = self.font.render(vehicles_text, True, (0, 0, 0))
        self.screen.blit(vehicles_surface, (10, 50))
        
        # Intersection information
        y_offset = 80
        for intersection_id, intersection in self.intersections.items():
            # Intersection ID
            int_text = f"Intersection {intersection_id}:"
            int_surface = self.font.render(int_text, True, (0, 0, 0))
            self.screen.blit(int_surface, (10, y_offset))
            
            # Queue lengths
            queue_text = f"  Queue: NS={intersection['queue_length']['NS']}, EW={intersection['queue_length']['EW']}"
            queue_surface = self.font.render(queue_text, True, (0, 0, 0))
            self.screen.blit(queue_surface, (10, y_offset + 20))
            
            # Waiting times
            wait_text = f"  Wait: NS={intersection['waiting_time']['NS']:.1f}s, EW={intersection['waiting_time']['EW']:.1f}s"
            wait_surface = self.font.render(wait_text, True, (0, 0, 0))
            self.screen.blit(wait_surface, (10, y_offset + 40))
            
            # Traffic light cycle
            ns_light = self.traffic_lights[f"{intersection_id}_NS"]
            cycle_text = f"  Cycle: {ns_light['cycle_length']:.1f}s, Green: {ns_light['green_time']:.1f}s"
            cycle_surface = self.font.render(cycle_text, True, (0, 0, 0))
            self.screen.blit(cycle_surface, (10, y_offset + 60))
            
            y_offset += 90
    
    def _get_simulation_time_str(self):
        """Get a formatted string of the simulation time."""
        # Convert simulation time to hours, minutes, seconds
        hours = self.simulation_time // 3600
        minutes = (self.simulation_time % 3600) // 60
        seconds = self.simulation_time % 60
        
        # Format as HH:MM:SS
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def generate_performance_report(self):
        """Generate a performance report for the simulation."""
        print("Generating performance report...")
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Collect metrics
        metrics = {
            "intersection_id": [],
            "avg_queue_length_ns": [],
            "avg_queue_length_ew": [],
            "avg_waiting_time_ns": [],
            "avg_waiting_time_ew": [],
            "avg_cycle_length": [],
            "avg_green_time_ns": [],
            "avg_green_time_ew": [],
            "throughput_ns": [],
            "throughput_ew": []
        }
        
        # Calculate metrics for each intersection
        for intersection_id, intersection in self.intersections.items():
            metrics["intersection_id"].append(intersection_id)
            metrics["avg_queue_length_ns"].append(intersection["queue_length"]["NS"])
            metrics["avg_queue_length_ew"].append(intersection["queue_length"]["EW"])
            metrics["avg_waiting_time_ns"].append(intersection["waiting_time"]["NS"])
            metrics["avg_waiting_time_ew"].append(intersection["waiting_time"]["EW"])
            
            # Traffic light metrics
            ns_light = self.traffic_lights[f"{intersection_id}_NS"]
            ew_light = self.traffic_lights[f"{intersection_id}_EW"]
            
            metrics["avg_cycle_length"].append(ns_light["cycle_length"])
            metrics["avg_green_time_ns"].append(ns_light["green_time"])
            metrics["avg_green_time_ew"].append(ew_light["green_time"])
            
            # Throughput (simplified calculation)
            metrics["throughput_ns"].append(intersection["volume"]["NS"] * 60)  # vehicles per hour
            metrics["throughput_ew"].append(intersection["volume"]["EW"] * 60)  # vehicles per hour
        
        # Create DataFrame
        df = pd.DataFrame(metrics)
        
        # Save to CSV
        csv_path = os.path.join(report_dir, "performance_metrics.csv")
        df.to_csv(csv_path, index=False)
        
        # Generate plots
        self._generate_performance_plots(df, report_dir)
        
        print(f"Performance report saved to {report_dir}")
        return report_dir
    
    def _generate_performance_plots(self, df, report_dir):
        """Generate performance plots from metrics."""
        # Queue length comparison
        plt.figure(figsize=(10, 6))
        x = np.arange(len(df["intersection_id"]))
        width = 0.35
        
        plt.bar(x - width/2, df["avg_queue_length_ns"], width, label="N-S")
        plt.bar(x + width/2, df["avg_queue_length_ew"], width, label="E-W")
        
        plt.xlabel("Intersection")
        plt.ylabel("Average Queue Length")
        plt.title("Average Queue Length by Intersection")
        plt.xticks(x, df["intersection_id"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "queue_length.png"))
        plt.close()
        
        # Waiting time comparison
        plt.figure(figsize=(10, 6))
        
        plt.bar(x - width/2, df["avg_waiting_time_ns"], width, label="N-S")
        plt.bar(x + width/2, df["avg_waiting_time_ew"], width, label="E-W")
        
        plt.xlabel("Intersection")
        plt.ylabel("Average Waiting Time (s)")
        plt.title("Average Waiting Time by Intersection")
        plt.xticks(x, df["intersection_id"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "waiting_time.png"))
        plt.close()
        
        # Throughput comparison
        plt.figure(figsize=(10, 6))
        
        plt.bar(x - width/2, df["throughput_ns"], width, label="N-S")
        plt.bar(x + width/2, df["throughput_ew"], width, label="E-W")
        
        plt.xlabel("Intersection")
        plt.ylabel("Throughput (vehicles/hour)")
        plt.title("Traffic Throughput by Intersection")
        plt.xticks(x, df["intersection_id"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "throughput.png"))
        plt.close()
        
        # Signal timing
        plt.figure(figsize=(10, 6))
        
        plt.bar(x, df["avg_cycle_length"], width, label="Cycle Length")
        plt.bar(x, df["avg_green_time_ns"], width, label="N-S Green Time", alpha=0.7)
        plt.bar(x, df["avg_green_time_ew"], width, label="E-W Green Time", alpha=0.5)
        
        plt.xlabel("Intersection")
        plt.ylabel("Time (s)")
        plt.title("Signal Timing by Intersection")
        plt.xticks(x, df["intersection_id"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "signal_timing.png"))
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create simulation
    simulation = CARLASimulationIntegration()
    
    # Initialize components
    simulation.initialize_components()
    
    # Set up simulation
    simulation.setup_simulation()
    
    # Run simulation
    simulation.run_simulation()
    
    # Generate performance report
    report_dir = simulation.generate_performance_report()
    
    print(f"Simulation completed. Performance report saved to {report_dir}")
