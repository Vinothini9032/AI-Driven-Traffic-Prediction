# AI-Driven Smart Traffic Congestion Model

This project implements an AI-driven smart traffic congestion model that integrates distributed image processing, YOLOv5 for vehicle detection, RNN-LSTM for traffic prediction, and fuzzy logic for adaptive signal control, all tested in a CARLA simulation environment.

## Overview

The system enhances autonomous smart traffic management (ASTM) by using artificial intelligence to reduce traffic congestion and increase efficiency. Key components include:

- **Distributed Image Processing**: Uses a multi-threaded approach for scalable image processing
- **YOLOv5 Vehicle Detection**: Provides accurate vehicle detection in traffic scenes
- **RNN-LSTM Traffic Prediction**: Forecasts traffic volume for the coming 12 hours
- **Fuzzy Logic Controller**: Implements adaptive signal control based on current and predicted traffic conditions
- **CARLA Simulation**: Tests the system in a simulated Chennai-based environment

## Performance Highlights

Based on simulation results, the system achieves:
- **Delay reduction**: Up to 70% compared to traditional fixed-timing systems
- **Traffic flow enhancement**: Up to 50% improvement in vehicles per minute
- **Queue length reduction**: Significant decrease in vehicle queues at intersections
- **Waiting time reduction**: Substantial decrease in average waiting time

## Project Structure

```
traffic_congestion_model/
├── data/                  # Data storage directory
├── docs/                  # Documentation files
├── models/                # Model storage directory
├── output/                # Output files and results
├── scripts/               # Implementation scripts
│   ├── distributed_image_processor.py     # Distributed image processing
│   ├── yolov5_vehicle_detector.py         # YOLOv5 vehicle detection
│   ├── rnn_lstm_predictor.py              # RNN-LSTM traffic prediction
│   ├── fuzzy_logic_controller.py          # Fuzzy logic adaptive control
│   ├── carla_simulation_integration.py    # CARLA simulation integration
│   └── model_tester.py                    # Model testing and evaluation
├── simulation/            # Simulation files
├── utils/                 # Utility functions
├── main.py                # Main entry point
└── README.md              # This file
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib
- OpenCV
- scikit-learn, scikit-fuzzy
- Pygame (for visualization)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/traffic-congestion-model.git
cd traffic-congestion-model
```

2. Install dependencies:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn opencv-python scikit-fuzzy pygame
```

## Usage

The system can be run in three different modes:

### 1. Simulation Mode

Run the complete simulation with either adaptive or fixed-timing control:

```
python main.py --mode simulation --output ./output/simulation --duration 3600 --adaptive
```

Parameters:
- `--output`: Directory to save simulation results
- `--duration`: Simulation duration in seconds
- `--adaptive`: Enable adaptive control (omit for fixed-timing)

### 2. Test Mode

Run comparative tests between adaptive and fixed-timing control:

```
python main.py --mode test --output ./output/test
```

This will generate a comprehensive evaluation report comparing the performance of adaptive and fixed-timing control strategies.

### 3. Component Mode

Run individual components for testing or demonstration:

```
python main.py --mode component --component predictor --output ./output/component
```

Available components:
- `processor`: Distributed image processor
- `detector`: YOLOv5 vehicle detector
- `predictor`: RNN-LSTM traffic predictor
- `controller`: Fuzzy logic controller

## Detailed Documentation

### Distributed Image Processing

The distributed image processor implements a multi-threaded approach to process traffic camera images efficiently. It divides the processing workload across multiple threads to achieve real-time performance.

### YOLOv5 Vehicle Detection

The YOLOv5 vehicle detector provides accurate detection of vehicles in traffic scenes. It uses a simplified implementation that focuses on the core detection functionality while maintaining accuracy.

### RNN-LSTM Traffic Prediction

The RNN-LSTM traffic predictor forecasts traffic volume for the coming 12 hours based on historical data. It uses a Long Short-Term Memory network to capture temporal patterns in traffic data.

### Fuzzy Logic Controller

The fuzzy logic controller implements adaptive signal control based on current traffic conditions and predictions. It uses fuzzy membership functions and rules to determine optimal signal timing.

### CARLA Simulation Integration

The CARLA simulation integration provides a realistic environment for testing the AI-driven traffic congestion model. It simulates traffic flow, intersections, and traffic lights in a Chennai-based environment.

## Evaluation

The system evaluation compares the performance of the AI-driven adaptive control against traditional fixed-timing control. Key metrics include:

- Average delay (seconds)
- Traffic flow (vehicles/minute)
- Queue length (vehicles)
- Waiting time (seconds)
- Cycle length (seconds)

The evaluation report includes detailed metrics, visualizations, and analysis of the performance improvements.

## Future Work

Potential areas for future enhancement include:
- Integration with real traffic camera feeds
- Expansion to larger urban networks
- Incorporation of pedestrian and cyclist detection
- Machine learning for parameter optimization
- Integration with connected vehicle technologies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CARLA Simulator for providing the simulation environment
- YOLOv5 for object detection capabilities
- The research community for advancements in traffic management systems
