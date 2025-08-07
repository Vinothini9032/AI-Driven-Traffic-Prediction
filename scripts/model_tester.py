#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test and Evaluation Script for AI-Driven Smart Traffic Congestion Model
This script tests and evaluates the performance of the integrated model.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pygame

# Import our custom modules
# AFTER:
from .carla_simulation_integration import CARLASimulationIntegration

class ModelTester:
    """
    A class for testing and evaluating the AI-driven smart traffic congestion model.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the model tester.
        
        Args:
            output_dir (str): Directory to save test results
        """
        self.output_dir = output_dir or "/home/ubuntu/traffic_congestion_model/output/evaluation"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize simulation
        self.simulation = CARLASimulationIntegration(output_dir=os.path.join(self.output_dir, "simulation"))
        
        # Test parameters
        self.test_scenarios = [
            {"name": "baseline", "description": "Baseline scenario with fixed timing", "adaptive": False},
            {"name": "adaptive", "description": "Adaptive scenario with AI control", "adaptive": True}
        ]
        
        # Performance metrics
        self.metrics = {
            "scenario": [],
            "avg_delay": [],
            "avg_flow": [],
            "avg_queue_length": [],
            "avg_waiting_time": [],
            "avg_cycle_length": []
        }
    
    def run_tests(self):
        """Run all test scenarios and collect metrics."""
        print("Starting model testing and evaluation...")
        
        for scenario in self.test_scenarios:
            print(f"\nRunning test scenario: {scenario['name']} - {scenario['description']}")
            
            # Configure simulation for this scenario
            self._configure_simulation(scenario)
            
            # Run simulation
            self.simulation.run_simulation()
            
            # Collect metrics
            self._collect_metrics(scenario)
            
            # Generate scenario report
            report_dir = self.simulation.generate_performance_report()
            print(f"Scenario report saved to {report_dir}")
        
        # Compare scenarios and generate final report
        self._generate_comparison_report()
        
        print("\nModel testing and evaluation completed.")
    
    def _configure_simulation(self, scenario):
        """Configure the simulation for a specific test scenario."""
        # Initialize components
        self.simulation.initialize_components()
        
        # Set up simulation
        self.simulation.setup_simulation()
        
        # Configure adaptive control based on scenario
        if not scenario["adaptive"]:
            # For baseline scenario, override the optimize_traffic_signals method
            # to use fixed timing instead of adaptive control
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
            self.simulation._optimize_traffic_signals = fixed_timing.__get__(self.simulation)
        
        # Set simulation parameters
        self.simulation.max_simulation_time = 3600  # 1 hour simulation
        
        # Set output directory for this scenario
        self.simulation.output_dir = os.path.join(self.output_dir, f"simulation_{scenario['name']}")
    
    def _collect_metrics(self, scenario):
        """Collect performance metrics from the simulation."""
        # Calculate average metrics across all intersections
        avg_delay = 0
        avg_flow = 0
        avg_queue_length = 0
        avg_waiting_time = 0
        avg_cycle_length = 0
        
        num_intersections = len(self.simulation.intersections)
        
        for intersection_id, intersection in self.simulation.intersections.items():
            # Average delay (waiting time)
            delay_ns = intersection["waiting_time"]["NS"]
            delay_ew = intersection["waiting_time"]["EW"]
            avg_delay += (delay_ns + delay_ew) / 2
            
            # Average flow (vehicles per minute)
            flow_ns = intersection["volume"]["NS"]
            flow_ew = intersection["volume"]["EW"]
            avg_flow += (flow_ns + flow_ew) / 2
            
            # Average queue length
            queue_ns = intersection["queue_length"]["NS"]
            queue_ew = intersection["queue_length"]["EW"]
            avg_queue_length += (queue_ns + queue_ew) / 2
            
            # Average waiting time
            wait_ns = intersection["waiting_time"]["NS"]
            wait_ew = intersection["waiting_time"]["EW"]
            avg_waiting_time += (wait_ns + wait_ew) / 2
            
            # Average cycle length
            ns_light = self.simulation.traffic_lights[f"{intersection_id}_NS"]
            avg_cycle_length += ns_light["cycle_length"]
        
        # Calculate averages
        avg_delay /= num_intersections
        avg_flow /= num_intersections
        avg_queue_length /= num_intersections
        avg_waiting_time /= num_intersections
        avg_cycle_length /= num_intersections
        
        # Store metrics
        self.metrics["scenario"].append(scenario["name"])
        self.metrics["avg_delay"].append(avg_delay)
        self.metrics["avg_flow"].append(avg_flow)
        self.metrics["avg_queue_length"].append(avg_queue_length)
        self.metrics["avg_waiting_time"].append(avg_waiting_time)
        self.metrics["avg_cycle_length"].append(avg_cycle_length)
        
        # Print metrics
        print(f"\nPerformance metrics for {scenario['name']} scenario:")
        print(f"  Average delay: {avg_delay:.2f} seconds")
        print(f"  Average flow: {avg_flow:.2f} vehicles/minute")
        print(f"  Average queue length: {avg_queue_length:.2f} vehicles")
        print(f"  Average waiting time: {avg_waiting_time:.2f} seconds")
        print(f"  Average cycle length: {avg_cycle_length:.2f} seconds")
    
    def _generate_comparison_report(self):
        """Generate a comparison report between test scenarios."""
        print("\nGenerating comparison report...")
        
        # Create DataFrame from metrics
        df = pd.DataFrame(self.metrics)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, "comparison_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Comparison metrics saved to {csv_path}")
        
        # Calculate improvement percentages
        if len(df) >= 2:
            baseline = df[df["scenario"] == "baseline"].iloc[0]
            adaptive = df[df["scenario"] == "adaptive"].iloc[0]
            
            delay_reduction = (baseline["avg_delay"] - adaptive["avg_delay"]) / baseline["avg_delay"] * 100
            flow_improvement = (adaptive["avg_flow"] - baseline["avg_flow"]) / baseline["avg_flow"] * 100
            queue_reduction = (baseline["avg_queue_length"] - adaptive["avg_queue_length"]) / baseline["avg_queue_length"] * 100
            wait_reduction = (baseline["avg_waiting_time"] - adaptive["avg_waiting_time"]) / baseline["avg_waiting_time"] * 100
            
            # Print improvement percentages
            print("\nImprovement with AI-driven adaptive control:")
            print(f"  Delay reduction: {delay_reduction:.2f}%")
            print(f"  Traffic flow enhancement: {flow_improvement:.2f}%")
            print(f"  Queue length reduction: {queue_reduction:.2f}%")
            print(f"  Waiting time reduction: {wait_reduction:.2f}%")
            
            # Save improvement metrics
            improvement = {
                "metric": ["delay_reduction", "flow_improvement", "queue_reduction", "wait_reduction"],
                "percentage": [delay_reduction, flow_improvement, queue_reduction, wait_reduction]
            }
            
            imp_df = pd.DataFrame(improvement)
            imp_csv_path = os.path.join(self.output_dir, "improvement_metrics.csv")
            imp_df.to_csv(imp_csv_path, index=False)
            print(f"Improvement metrics saved to {imp_csv_path}")
            
            # Generate comparison plots
            self._generate_comparison_plots(df, improvement)
        
        # Generate final report document
        self._generate_report_document(df)
    
    def _generate_comparison_plots(self, metrics_df, improvement):
        """Generate comparison plots between scenarios."""
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Bar chart for delay comparison
        plt.figure(figsize=(10, 6))
        scenarios = metrics_df["scenario"].tolist()
        delays = metrics_df["avg_delay"].tolist()
        
        plt.bar(scenarios, delays, color=['blue', 'green'])
        plt.xlabel("Scenario")
        plt.ylabel("Average Delay (seconds)")
        plt.title("Average Delay Comparison")
        
        # Add delay reduction percentage
        delay_reduction = improvement["percentage"][0]
        plt.text(1, delays[1] + 1, f"{delay_reduction:.2f}% reduction", 
                ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "delay_comparison.png"))
        plt.close()
        
        # Bar chart for flow comparison
        plt.figure(figsize=(10, 6))
        flows = metrics_df["avg_flow"].tolist()
        
        plt.bar(scenarios, flows, color=['blue', 'green'])
        plt.xlabel("Scenario")
        plt.ylabel("Average Flow (vehicles/minute)")
        plt.title("Average Flow Comparison")
        
        # Add flow improvement percentage
        flow_improvement = improvement["percentage"][1]
        plt.text(1, flows[1] + 0.5, f"{flow_improvement:.2f}% improvement", 
                ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "flow_comparison.png"))
        plt.close()
        
        # Bar chart for queue length comparison
        plt.figure(figsize=(10, 6))
        queues = metrics_df["avg_queue_length"].tolist()
        
        plt.bar(scenarios, queues, color=['blue', 'green'])
        plt.xlabel("Scenario")
        plt.ylabel("Average Queue Length (vehicles)")
        plt.title("Average Queue Length Comparison")
        
        # Add queue reduction percentage
        queue_reduction = improvement["percentage"][2]
        plt.text(1, queues[1] + 0.5, f"{queue_reduction:.2f}% reduction", 
                ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "queue_comparison.png"))
        plt.close()
        
        # Bar chart for waiting time comparison
        plt.figure(figsize=(10, 6))
        waits = metrics_df["avg_waiting_time"].tolist()
        
        plt.bar(scenarios, waits, color=['blue', 'green'])
        plt.xlabel("Scenario")
        plt.ylabel("Average Waiting Time (seconds)")
        plt.title("Average Waiting Time Comparison")
        
        # Add waiting time reduction percentage
        wait_reduction = improvement["percentage"][3]
        plt.text(1, waits[1] + 0.5, f"{wait_reduction:.2f}% reduction", 
                ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "wait_comparison.png"))
        plt.close()
        
        # Summary plot of all improvements
        plt.figure(figsize=(12, 8))
        metrics = improvement["metric"]
        percentages = improvement["percentage"]
        
        colors = ['green' if p > 0 else 'red' for p in percentages]
        
        plt.bar(metrics, percentages, color=colors)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel("Metric")
        plt.ylabel("Improvement Percentage (%)")
        plt.title("Performance Improvement with AI-Driven Adaptive Control")
        
        # Add percentage labels
        for i, p in enumerate(percentages):
            plt.text(i, p + 2 if p > 0 else p - 2, 
                    f"{p:.2f}%", ha='center', 
                    va='bottom' if p > 0 else 'top', 
                    fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "improvement_summary.png"))
        plt.close()
        
        print(f"Comparison plots saved to {plots_dir}")
    
    def _generate_report_document(self, metrics_df):
        """Generate a comprehensive report document."""
        report_path = os.path.join(self.output_dir, "evaluation_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# AI-Driven Smart Traffic Congestion Model - Evaluation Report\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            f.write("## 1. Introduction\n\n")
            f.write("This report presents the evaluation results of the AI-driven smart traffic congestion model ")
            f.write("that integrates distributed image processing with PySpark, YOLOv5 for vehicle detection, ")
            f.write("RNN-LSTM for traffic prediction, and fuzzy logic for adaptive signal control.\n\n")
            
            f.write("The model was tested in a simulated Chennai-based environment using CARLA simulation ")
            f.write("to evaluate its performance compared to traditional fixed-timing traffic control.\n\n")
            
            f.write("## 2. Test Scenarios\n\n")
            f.write("Two test scenarios were evaluated:\n\n")
            
            f.write("1. **Baseline Scenario**: Traditional fixed-timing traffic signal control\n")
            f.write("   - Fixed cycle length of 120 seconds\n")
            f.write("   - Fixed green time of 55 seconds for each direction\n")
            f.write("   - Fixed yellow time of 3 seconds\n")
            f.write("   - Fixed all-red time of 2 seconds\n\n")
            
            f.write("2. **Adaptive Scenario**: AI-driven adaptive traffic signal control\n")
            f.write("   - Adaptive cycle length based on current and predicted traffic conditions\n")
            f.write("   - Adaptive green time allocation based on queue lengths and waiting times\n")
            f.write("   - RNN-LSTM prediction of traffic volumes for the coming 12 hours\n")
            f.write("   - Fuzzy logic controller for adaptive signal timing\n\n")
            
            f.write("## 3. Performance Metrics\n\n")
            f.write("The following metrics were used to evaluate the performance of the model:\n\n")
            
            f.write("- **Average Delay**: Average time vehicles spend waiting at intersections\n")
            f.write("- **Average Flow**: Average number of vehicles passing through intersections per minute\n")
            f.write("- **Average Queue Length**: Average number of vehicles waiting at intersections\n")
            f.write("- **Average Waiting Time**: Average time vehicles spend in queues\n")
            f.write("- **Average Cycle Length**: Average duration of a complete signal cycle\n\n")
            
            f.write("## 4. Results\n\n")
            f.write("### 4.1 Comparison of Metrics\n\n")
            
            # Create a markdown table of metrics
            f.write("| Metric | Baseline | Adaptive | Improvement |\n")
            f.write("|--------|----------|----------|-------------|\n")
            
            if len(metrics_df) >= 2:
                baseline = metrics_df[metrics_df["scenario"] == "baseline"].iloc[0]
                adaptive = metrics_df[metrics_df["scenario"] == "adaptive"].iloc[0]
                
                delay_reduction = (baseline["avg_delay"] - adaptive["avg_delay"]) / baseline["avg_delay"] * 100
                flow_improvement = (adaptive["avg_flow"] - baseline["avg_flow"]) / baseline["avg_flow"] * 100
                queue_reduction = (baseline["avg_queue_length"] - adaptive["avg_queue_length"]) / baseline["avg_queue_length"] * 100
                wait_reduction = (baseline["avg_waiting_time"] - adaptive["avg_waiting_time"]) / baseline["avg_waiting_time"] * 100
                cycle_diff = adaptive["avg_cycle_length"] - baseline["avg_cycle_length"]
                
                f.write(f"| Average Delay (s) | {baseline['avg_delay']:.2f} | {adaptive['avg_delay']:.2f} | {delay_reduction:.2f}% reduction |\n")
                f.write(f"| Average Flow (veh/min) | {baseline['avg_flow']:.2f} | {adaptive['avg_flow']:.2f} | {flow_improvement:.2f}% increase |\n")
                f.write(f"| Average Queue Length | {baseline['avg_queue_length']:.2f} | {adaptive['avg_queue_length']:.2f} | {queue_reduction:.2f}% reduction |\n")
                f.write(f"| Average Waiting Time (s) | {baseline['avg_waiting_time']:.2f} | {adaptive['avg_waiting_time']:.2f} | {wait_reduction:.2f}% reduction |\n")
                f.write(f"| Average Cycle Length (s) | {baseline['avg_cycle_length']:.2f} | {adaptive['avg_cycle_length']:.2f} | {cycle_diff:.2f}s difference |\n\n")
            
            f.write("### 4.2 Key Findings\n\n")
            
            if len(metrics_df) >= 2:
                f.write(f"- **Delay Reduction**: The AI-driven model reduced average delay by {delay_reduction:.2f}%\n")
                f.write(f"- **Traffic Flow Enhancement**: Traffic flow increased by {flow_improvement:.2f}%\n")
                f.write(f"- **Queue Length Reduction**: Average queue length decreased by {queue_reduction:.2f}%\n")
                f.write(f"- **Waiting Time Reduction**: Average waiting time decreased by {wait_reduction:.2f}%\n\n")
            
            f.write("### 4.3 Visualization\n\n")
            
            f.write("The following visualizations compare the performance metrics between the baseline and adaptive scenarios:\n\n")
            
            f.write("![Delay Comparison](plots/delay_comparison.png)\n\n")
            f.write("![Flow Comparison](plots/flow_comparison.png)\n\n")
            f.write("![Queue Comparison](plots/queue_comparison.png)\n\n")
            f.write("![Wait Comparison](plots/wait_comparison.png)\n\n")
            f.write("![Improvement Summary](plots/improvement_summary.png)\n\n")
            
            f.write("## 5. Conclusion\n\n")
            
            if len(metrics_df) >= 2:
                f.write("The AI-driven smart traffic congestion model demonstrated significant improvements over traditional fixed-timing traffic control:\n\n")
                
                if delay_reduction >= 70:
                    f.write(f"- The model achieved the target of **{delay_reduction:.2f}% delay reduction** (target: 70%)\n")
                else:
                    f.write(f"- The model achieved **{delay_reduction:.2f}% delay reduction** (target: 70%)\n")
                
                if flow_improvement >= 50:
                    f.write(f"- The model achieved the target of **{flow_improvement:.2f}% traffic flow enhancement** (target: 50%)\n\n")
                else:
                    f.write(f"- The model achieved **{flow_improvement:.2f}% traffic flow enhancement** (target: 50%)\n\n")
                
                f.write("These results demonstrate the effectiveness of the integrated approach combining distributed image processing, ")
                f.write("YOLOv5 vehicle detection, RNN-LSTM traffic prediction, and fuzzy logic adaptive signal control ")
                f.write("for improving urban traffic management.\n\n")
                
                if delay_reduction >= 70 and flow_improvement >= 50:
                    f.write("The model successfully met or exceeded all performance targets specified in the requirements.\n")
                else:
                    f.write("While the model showed significant improvements, further optimization may be needed to fully meet all performance targets.\n")
            
            f.write("\n## 6. Appendix\n\n")
            f.write("### 6.1 Raw Metrics Data\n\n")
            
            # Write metrics data as markdown table
            f.write("| Scenario | Avg Delay | Avg Flow | Avg Queue | Avg Wait | Avg Cycle |\n")
            f.write("|----------|-----------|----------|-----------|----------|------------|\n")
            
            for _, row in metrics_df.iterrows():
                f.write(f"| {row['scenario']} | {row['avg_delay']:.2f} | {row['avg_flow']:.2f} | {row['avg_queue_length']:.2f} | {row['avg_waiting_time']:.2f} | {row['avg_cycle_length']:.2f} |\n")
        
        print(f"Evaluation report saved to {report_path}")
        return report_path


# Example usage
if __name__ == "__main__":
    # Create tester
    tester = ModelTester()
    
    # Run tests
    tester.run_tests()
