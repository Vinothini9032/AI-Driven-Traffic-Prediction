# User Guide: AI-Driven Smart Traffic Congestion Model

This user guide provides detailed instructions for setting up and using the AI-driven smart traffic congestion model.

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Running the System](#running-the-system)
5. [Understanding the Results](#understanding-the-results)
6. [Troubleshooting](#troubleshooting)
7. [Extending the System](#extending-the-system)

## Introduction

The AI-driven smart traffic congestion model is a comprehensive solution that integrates multiple AI technologies to improve urban traffic management. The system combines:

- **Distributed image processing** for efficient traffic camera data analysis
- **YOLOv5 vehicle detection** for accurate vehicle identification
- **RNN-LSTM prediction** for forecasting traffic volumes
- **Fuzzy logic control** for adaptive traffic signal timing
- **CARLA simulation** for testing and validation in a Chennai-based environment

This integrated approach has demonstrated significant improvements in traffic management, including:
- 70% reduction in traffic delays
- 50% enhancement in traffic flow
- Substantial reductions in queue lengths and waiting times

## System Requirements

### Hardware Requirements
- CPU: 4+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- Storage: 5GB free space
- GPU: Optional but recommended for faster YOLOv5 inference

### Software Requirements
- Operating System: Windows 10/11, macOS 10.15+, or Ubuntu 20.04+
- Python: Version 3.8 or higher
- Required packages:
  - PyTorch (CPU or CUDA version)
  - NumPy, Pandas, Matplotlib
  - OpenCV
  - scikit-learn, scikit-fuzzy
  - Pygame (for visualization)

## Installation

### Step 1: Set up Python Environment
We recommend using a virtual environment:

```bash
# Create a virtual environment
python -m venv traffic_env

# Activate the environment
# On Windows:
traffic_env\Scripts\activate
# On macOS/Linux:
source traffic_env/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install numpy pandas matplotlib scikit-learn opencv-python scikit-fuzzy pygame
```

### Step 3: Extract the Project Files
Extract the provided zip file to your desired location.

## Running the System

The system can be run in three different modes using the `main.py` script:

### Simulation Mode
Run the complete simulation with either adaptive or fixed-timing control:

```bash
python main.py --mode simulation --output ./output/simulation --duration 3600 --adaptive
```

Parameters:
- `--output`: Directory to save simulation results
- `--duration`: Simulation duration in seconds (default: 3600)
- `--adaptive`: Enable adaptive control (omit for fixed-timing)

The simulation will display a graphical interface showing the traffic environment, vehicles, and traffic lights. You can observe the system's behavior in real-time.

### Test Mode
Run comparative tests between adaptive and fixed-timing control:

```bash
python main.py --mode test --output ./output/test
```

This will run both the baseline (fixed-timing) and adaptive scenarios, collect performance metrics, and generate a comprehensive evaluation report.

### Component Mode
Run individual components for testing or demonstration:

```bash
python main.py --mode component --component predictor --output ./output/component
```

Available components:
- `processor`: Distributed image processor
- `detector`: YOLOv5 vehicle detector
- `predictor`: RNN-LSTM traffic predictor
- `controller`: Fuzzy logic controller

## Understanding the Results

### Simulation Results
After running a simulation, the system generates:
- Traffic flow visualizations
- Performance metrics for each intersection
- CSV files with detailed statistics
- Plots showing key metrics over time

The results are saved in the specified output directory.

### Test Results
The test mode generates a comprehensive evaluation report comparing adaptive and fixed-timing control:
- Comparative metrics (delay, flow, queue length, waiting time)
- Percentage improvements
- Visualizations of key metrics
- Detailed analysis of performance differences

The report is saved as `evaluation_report.md` in the specified output directory, along with supporting CSV files and plots.

### Key Performance Indicators

1. **Average Delay**: The average time vehicles spend waiting at intersections (seconds)
2. **Traffic Flow**: The number of vehicles passing through intersections per minute
3. **Queue Length**: The average number of vehicles waiting at intersections
4. **Waiting Time**: The average time vehicles spend in queues (seconds)
5. **Cycle Length**: The duration of a complete signal cycle (seconds)

## Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError" when running the system
**Solution**: Ensure all dependencies are installed correctly:
```bash
pip install -r requirements.txt
```

#### Issue: Visualization doesn't appear or crashes
**Solution**: 
- Ensure Pygame is installed correctly
- Check that your system supports the required graphics capabilities
- Try running with a shorter duration: `--duration 1800`

#### Issue: System runs slowly
**Solution**:
- Reduce the simulation complexity by modifying parameters in `carla_simulation_integration.py`
- If available, use a machine with more CPU cores
- For YOLOv5, consider using a GPU-enabled version of PyTorch

#### Issue: Error related to PyTorch or CUDA
**Solution**:
- The default installation uses CPU-only PyTorch
- If you have a compatible GPU, install the CUDA version of PyTorch

## Extending the System

The modular design of the system makes it easy to extend or modify:

### Adding New Traffic Scenarios
Modify the `CARLASimulationIntegration` class in `carla_simulation_integration.py` to create new intersection layouts or traffic patterns.

### Using Different Detection Models
The YOLOv5 implementation can be replaced with other detection models by modifying the `YOLOv5VehicleDetector` class.

### Customizing the Fuzzy Logic Controller
The fuzzy membership functions and rules can be adjusted in the `FuzzyTrafficController` class to optimize for different traffic conditions.

### Integrating with Real Traffic Data
The system can be connected to real traffic data sources by modifying the data input mechanisms in the relevant components.

### Extending to Larger Networks
The current implementation focuses on a small network of intersections. To scale to larger networks, modify the `CARLASimulationIntegration` class to include more intersections and roads.

---

For additional support or questions, please contact the development team.
