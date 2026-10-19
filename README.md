# Master Thesis Code - MPC to DNN Path Following

This project implements a Model Predictive Controller (MPC) for path following and trains a Deep Neural Network (DNN) to mimic its behavior, enabling fast real-time control.

## Project Structure

```
.
├── src/                          # Source code
│   ├── models/                   # Dynamical models
│   │   └── unicycle.py          # Kinematic unicycle model
│   ├── controllers/             # Control algorithms
│   │   ├── mpc_controller.py   # MPC controller implementation
│   │   └── dnn_controller.py    # DNN controller and neural network
│   ├── training/                # Training pipeline
│   │   └── trainer.py          # DNN training utilities
│   └── utils/                   # Utility functions
│       ├── reference.py        # Reference trajectory generation
│       ├── data_collection.py  # MPC data collection
│       └── visualization.py    # Plotting and visualization
├── scripts/                     # Executable scripts
│   ├── train_mpc_to_dnn.py     # Main training pipeline
│   ├── archive_init.py         # Original monolithic Init.py
│   ├── standalone_lqr_test.py  # LQR performance benchmark
│   ├── plot_fig44.py            # Thesis figure generation
│   └── main.py                  # Placeholder main script
├── tests/                       # Unit tests
│   └── test_unicycle.py        # Unicycle model tests
├── data/                        # Training data
├── models/                      # Saved model weights
├── outputs/                     # Generated plots and figures
└── my_optimizers/              # OpEn optimizer builds (generated)
```

## Quick Start

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 2. Run Unit Tests

```bash
python tests/test_unicycle.py
```

### 3. Train MPC to DNN

```bash
python scripts/train_mpc_to_dnn.py
```

This will:
- Build the MPC optimizer using OpEn
- Collect 8000 training samples from MPC
- Train a neural network to mimic MPC
- Evaluate the DNN controller on multiple initial conditions
- Generate performance visualizations

## Components

### Models

- **Unicycle Model**: Kinematic unicycle with state `[x, y, theta]` and control `[v, omega]`
- Supports both numpy and CasADi symbolic dynamics

### Controllers

- **MPC Controller**: Solves optimal control problem over prediction horizon N=70
- **DNN Controller**: Neural network approximation of MPC with 128×128×64×32 architecture

### Training Pipeline

- Data collection from MPC on elliptical reference trajectory
- Neural network training with early stopping and learning rate scheduling
- Input normalization and output constraint handling

### Utilities

- Reference trajectory generation (elliptical path)
- Data collection with noise injection
- Comprehensive visualization of training and results

## Dependencies

- Python 3.x
- numpy
- torch
- opengen
- casadi
- matplotlib
- scipy
- tqdm

## Output Files

Training generates:
- `data/mpc_dataset.npz` - Training dataset
- `models/best_mpc_dnn_model.pth` - Best model during training
- `models/final_mpc_dnn_model.pth` - Final trained model
- `outputs/training_curves.png` - Training progress plots
- `outputs/dnn_results_comprehensive.png` - Performance analysis

## Architecture

The MPC uses an augmented state `[x, y, theta, theta_p]` where:
- `x, y, theta`: Robot pose
- `theta_p`: Path parameter (progress along reference)

Control inputs `[s, omega, v]`:
- `s`: Forward speed
- `omega`: Angular velocity
- `v`: Path parameter velocity

## References

This implementation follows the methodology for learning MPC policies using deep neural networks for real-time path following control.
