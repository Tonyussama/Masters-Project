"""
Visualize and test the Colab-trained DNN model.

This script:
1. Loads the Colab-trained model
2. Runs closed-loop simulation on the ellipse
3. Visualizes trajectory and tracking error
4. Computes performance metrics (RMS error, max error, lap time)
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.dnn_controller import MPCNeuralNetwork, DNNController
from controllers.mpc_controller import MPCController
from utils.reference import generate_ellipse_reference, find_projection_index


def simulate_trajectory(controller, initial_state, ts, steps=300):
    """Simulate closed-loop trajectory.
    
    Args:
        controller: DNNController instance
        initial_state: Initial geometric state [x, y, theta]
        ts: Time step
        steps: Number of simulation steps
    
    Returns:
        trajectory: Array of augmented states
        controls: Array of control actions
        fallback_used: Array indicating when MPC fallback was used
    """
    trajectory = np.zeros((steps + 1, 4))
    controls = np.zeros((steps, 3))
    fallback_used = np.zeros(steps, dtype=bool)
    
    # Convert initial geometric state to augmented state
    idx_closest = find_projection_index(initial_state, controller.collector.x_ref_array, controller.collector.y_ref_array)
    theta_p_initial = controller.t_param[idx_closest]
    aug_current = np.concatenate([initial_state, [theta_p_initial]])
    trajectory[0] = aug_current
    
    for i in range(steps):
        u = controller.get_control(aug_current)
        s, w, v_path = u[0], u[1], u[2]
        controls[i] = u
        
        # Check if fallback was used (via controller stats)
        stats_before = controller.get_fallback_stats()
        fallback_used[i] = stats_before['fallback_count'] > sum(fallback_used[:i])
        
        # Apply dynamics
        aug_current = np.array([
            aug_current[0] + ts * s * np.cos(aug_current[2]),
            aug_current[1] + ts * s * np.sin(aug_current[2]),
            aug_current[2] + ts * w,
            aug_current[3] + ts * v_path
        ])
        # Wrap angle and path parameter
        aug_current[2] = (aug_current[2] + np.pi) % (2 * np.pi) - np.pi
        aug_current[3] = aug_current[3] % (2 * np.pi)
        trajectory[i + 1] = aug_current
    
    return trajectory, controls, fallback_used


def compute_tracking_error(trajectory, x_ref_array, y_ref_array, t_param):
    """Compute RMS and max tracking error for a trajectory.
    
    Args:
        trajectory: Array of augmented states
        x_ref_array: Reference x positions
        y_ref_array: Reference y positions
        t_param: Path parameter array
    
    Returns:
        rms_error: RMS tracking error in meters
        max_error: Maximum tracking error in meters
        errors: Array of per-step errors
    """
    errors = []
    for aug_state in trajectory:
        # Find closest point on reference path
        idx = find_projection_index(aug_state[:2], x_ref_array, y_ref_array)
        x_path = x_ref_array[idx]
        y_path = y_ref_array[idx]
        
        # Compute cross-track error
        error = np.sqrt((aug_state[0] - x_path)**2 + (aug_state[1] - y_path)**2)
        errors.append(error)
    
    errors = np.array(errors)
    rms_error = np.sqrt(np.mean(errors**2))
    max_error = np.max(errors)
    
    return rms_error, max_error, errors


def plot_results(trajectory, x_ref_array, y_ref_array, initial_state, ts, fallback_used=None, save_path='outputs/colab_model_results.png'):
    """Plot trajectory and tracking error.
    
    Args:
        trajectory: Array of augmented states
        x_ref_array: Reference x positions
        y_ref_array: Reference y positions
        initial_state: Initial state
        ts: Time step
        fallback_used: Array indicating when MPC fallback was used
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Trajectory
    axes[0, 0].plot(x_ref_array, y_ref_array, 'b--', label='Reference', linewidth=2, alpha=0.7)
    axes[0, 0].plot(trajectory[:, 0], trajectory[:, 1], 'r-', label='DNN Trajectory', linewidth=2)
    axes[0, 0].plot(initial_state[0], initial_state[1], 'go', markersize=10, label='Start')
    axes[0, 0].plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8)
    axes[0, 0].plot(trajectory[-1, 0], trajectory[-1, 1], 'rs', markersize=10, label='End')
    
    # Mark fallback points if available
    if fallback_used is not None and np.any(fallback_used):
        fallback_indices = np.where(fallback_used)[0]
        axes[0, 0].scatter(trajectory[fallback_indices, 0], trajectory[fallback_indices, 1], 
                          c='orange', s=50, marker='x', label='MPC Fallback', zorder=5)
    
    axes[0, 0].set_xlabel('X [m]')
    axes[0, 0].set_ylabel('Y [m]')
    axes[0, 0].set_title('Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Plot 2: Tracking error over time
    rms_error, max_error, errors = compute_tracking_error(trajectory, x_ref_array, y_ref_array, None)
    time = np.arange(len(errors)) * ts
    axes[0, 1].plot(time, errors * 1000, 'r-', linewidth=2)
    axes[0, 1].axhline(y=rms_error * 1000, color='g', linestyle='--', label=f'RMS: {rms_error*1000:.2f} mm')
    axes[0, 1].axhline(y=max_error * 1000, color='orange', linestyle='--', label=f'Max: {max_error*1000:.2f} mm')
    
    # Mark fallback regions
    if fallback_used is not None and np.any(fallback_used):
        fallback_indices = np.where(fallback_used)[0]
        axes[0, 1].scatter(time[fallback_indices], errors[fallback_indices] * 1000, 
                          c='orange', s=30, marker='x', label='MPC Fallback', zorder=5)
    
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Tracking Error [mm]')
    axes[0, 1].set_title('Tracking Error Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Control inputs
    time = np.arange(len(trajectory) - 1) * ts
    axes[1, 0].plot(time, trajectory[:-1, 2], 'g-', label='Heading θ [rad]', linewidth=2)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Heading [rad]')
    axes[1, 0].set_title('Robot Heading Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Path parameter (progress along track)
    axes[1, 1].plot(time, trajectory[:-1, 3], 'purple', label='Path Parameter θ_p', linewidth=2)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Path Parameter [rad]')
    axes[1, 1].set_title('Progress Along Track')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()


def main():
    print("="*70)
    print(" Testing Colab-Trained DNN Model")
    print("="*70)
    
    # =========================================
    # Setup Reference Trajectory
    # =========================================
    total_points = 10000
    t_param, x_ref_array, y_ref_array, theta_ref_array = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    print(f"\n[1/5] Reference trajectory created: {len(x_ref_array)} points")
    
    # =========================================
    # Load Trained Model
    # =========================================
    model_path = "models/Colab Trained Model.pth"
    print(f"\n[2/5] Loading model from {model_path}")
    
    ref_feature_dim = 42  # From data collection (14 points × 3 features)
    model = MPCNeuralNetwork(
        ref_feature_dim=ref_feature_dim,
        hidden_dims=[1024, 1024, 512, 256, 128],
        output_dim=2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"      Model loaded on {device}")
    
    # =========================================
    # Setup Controller with MPC Fallback
    # =========================================
    from utils.data_collection import MPCDataCollector
    
    # Build MPC for fallback
    nu = 3
    nx_aug = 4
    N = 70
    ts = 0.03
    q = 100
    qtheta = 10
    r_control = 0.1
    r_v = 0.001
    v_ref = 0.2
    
    print("\n[3/5] Building MPC for fallback...")
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    solver = mpc.build_optimizer(build_dir="my_optimizers")
    print("      MPC optimizer ready for fallback!")
    
    # Create collector
    collector = MPCDataCollector(solver, t_param, x_ref_array, y_ref_array, 
                                 theta_ref_array, N, nu, ts, v_ref)
    
    # Create controller with MPC fallback
    controller = DNNController(model, collector, t_param, v_ref=v_ref,
                              mpc_solver=solver, cross_track_threshold=0.5)
    print("      Controller with MPC fallback enabled")
    
    # =========================================
    # Test Full Lap Starting from Right Pointing Up
    # =========================================
    print("\n[4/5] Simulating full lap starting from right pointing up...")
    
    # Right side of ellipse (x ≈ 0.2), pointing up (theta = π/2)
    initial_state = np.array([0.2, 0.0, np.pi/2])
    
    # Calculate steps needed for full lap
    # Lap time ≈ 2π / v_ref ≈ 6.28 / 0.2 ≈ 31.4 seconds
    # Steps = 31.4 / 0.03 ≈ 1047 steps
    steps_for_lap = int(2 * np.pi / v_ref / ts) + 100  # Add buffer
    
    print(f"  Initial state: x={initial_state[0]:.2f}, y={initial_state[1]:.2f}, θ={initial_state[2]:.2f} rad")
    print(f"  Simulating {steps_for_lap} steps (~{steps_for_lap * ts:.1f} seconds)")
    
    traj, controls, fallback_used = simulate_trajectory(controller, initial_state, ts, steps=steps_for_lap)
    
    rms_error, max_error, errors = compute_tracking_error(traj, x_ref_array, y_ref_array, t_param)
    
    print(f"  RMS error: {rms_error*1000:.2f} mm")
    print(f"  Max error: {max_error*1000:.2f} mm")
    print(f"  MPC fallback used: {np.sum(fallback_used)} times ({np.sum(fallback_used)/len(fallback_used)*100:.1f}% of steps)")
    
    # =========================================
    # Summary Statistics
    # =========================================
    print("\n[5/5] Performance Summary:")
    print(f"  RMS error: {rms_error*1000:.2f} mm")
    print(f"  Max error: {max_error*1000:.2f} mm")
    print(f"  Lap time: {len(traj) * ts:.2f} seconds")
    print(f"  MPC fallback rate: {np.sum(fallback_used)/len(fallback_used)*100:.1f}%")
    
    # =========================================
    # Visualize Results
    # =========================================
    plot_results(traj, x_ref_array, y_ref_array, initial_state, ts, fallback_used=fallback_used)
    
    print("\n" + "="*70)
    print(" Testing Complete!")
    print("="*70)
    print(f"\nVisualization saved to: outputs/colab_model_results.png")
    print("\nLegend:")
    print("  - Blue dashed: Reference path")
    print("  - Red solid: DNN trajectory")
    print("  - Green circle: Start position")
    print("  - Red square: End position")
    print("  - Orange X: MPC fallback points (if any)")


if __name__ == "__main__":
    main()
