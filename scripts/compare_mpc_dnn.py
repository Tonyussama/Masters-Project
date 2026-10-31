"""
Compare MPC and DNN controllers side-by-side.

This script:
1. Simulates both MPC and DNN on the same initial condition
2. Creates side-by-side visualizations
3. Compares performance metrics
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.mpc_controller import MPCController
from controllers.dnn_controller import MPCNeuralNetwork, DNNController
from utils.reference import generate_ellipse_reference, find_projection_index
from utils.data_collection import MPCDataCollector


def simulate_mpc(solver, aug_state, ts, steps, t_param, x_ref_array, y_ref_array, theta_ref_array):
    """Simulate MPC trajectory.
    
    Args:
        solver: MPC solver
        aug_state: Initial augmented state
        ts: Time step
        steps: Number of steps
        t_param: Path parameter array
        x_ref_array: Reference x positions
        y_ref_array: Reference y positions
        theta_ref_array: Reference heading
    
    Returns:
        trajectory: Array of augmented states
        controls: Array of control actions
    """
    trajectory = np.zeros((steps + 1, 4))
    controls = np.zeros((steps, 3))
    trajectory[0] = aug_state
    
    for i in range(steps):
        # Get reference horizon
        theta_p = aug_state[3]
        x_ref = np.interp(theta_p + np.arange(70) * ts * 0.2, t_param, x_ref_array)
        y_ref = np.interp(theta_p + np.arange(70) * ts * 0.2, t_param, y_ref_array)
        theta_ref = np.interp(theta_p + np.arange(70) * ts * 0.2, t_param, np.unwrap(theta_ref_array))
        theta_ref = (theta_ref + np.pi) % (2 * np.pi) - np.pi
        
        # Solve MPC
        p_vector = np.concatenate([aug_state, x_ref, y_ref, theta_ref])
        u_init = [0.01] * (3 * 70)
        result = solver.run(p=p_vector, initial_guess=u_init)
        
        if result.exit_status == 'Converged':
            u = result.solution[0:3]
        else:
            # Fallback to previous control if MPC fails
            u = controls[max(0, i-1)]
        
        controls[i] = u
        
        # Apply dynamics
        aug_current = np.array([
            aug_state[0] + ts * u[0] * np.cos(aug_state[2]),
            aug_state[1] + ts * u[0] * np.sin(aug_state[2]),
            aug_state[2] + ts * u[1],
            aug_state[3] + ts * u[2]
        ])
        aug_current[2] = (aug_current[2] + np.pi) % (2 * np.pi) - np.pi
        aug_current[3] = aug_current[3] % (2 * np.pi)
        aug_state = aug_current
        trajectory[i + 1] = aug_state
    
    return trajectory, controls


def simulate_dnn(controller, aug_state, ts, steps):
    """Simulate DNN trajectory.
    
    Args:
        controller: DNNController instance
        aug_state: Initial augmented state
        ts: Time step
        steps: Number of steps
    
    Returns:
        trajectory: Array of augmented states
        controls: Array of control actions
    """
    trajectory = np.zeros((steps + 1, 4))
    controls = np.zeros((steps, 3))
    trajectory[0] = aug_state
    
    for i in range(steps):
        u = controller.get_control(aug_state)
        controls[i] = u
        
        # Apply dynamics
        aug_current = np.array([
            aug_state[0] + ts * u[0] * np.cos(aug_state[2]),
            aug_state[1] + ts * u[0] * np.sin(aug_state[2]),
            aug_state[2] + ts * u[1],
            aug_state[3] + ts * u[2]
        ])
        aug_current[2] = (aug_current[2] + np.pi) % (2 * np.pi) - np.pi
        aug_current[3] = aug_current[3] % (2 * np.pi)
        aug_state = aug_current
        trajectory[i + 1] = aug_state
    
    return trajectory, controls


def compute_tracking_error(trajectory, x_ref_array, y_ref_array):
    """Compute tracking error."""
    errors = []
    for aug_state in trajectory[:-1]:  # Exclude last state (same as first in closed loop)
        idx = find_projection_index(aug_state[:2], x_ref_array, y_ref_array)
        x_path = x_ref_array[idx]
        y_path = y_ref_array[idx]
        error = np.sqrt((aug_state[0] - x_path)**2 + (aug_state[1] - y_path)**2)
        errors.append(error)
    errors = np.array(errors)
    rms_error = np.sqrt(np.mean(errors**2))
    max_error = np.max(errors)
    return rms_error, max_error, errors


def plot_comparison(mpc_traj, dnn_traj, mpc_errors, dnn_errors, x_ref, y_ref, initial_state, ts, save_path='outputs/mpc_dnn_comparison.png'):
    """Plot MPC vs DNN comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    time_mpc = np.arange(len(mpc_errors)) * ts
    time_dnn = np.arange(len(dnn_errors)) * ts
    
    # Plot 1: Trajectories
    axes[0, 0].plot(x_ref, y_ref, 'k--', label='Reference', linewidth=2, alpha=0.5)
    axes[0, 0].plot(mpc_traj[:, 0], mpc_traj[:, 1], 'b-', label='MPC', linewidth=2, alpha=0.7)
    axes[0, 0].plot(dnn_traj[:, 0], dnn_traj[:, 1], 'r-', label='DNN', linewidth=2, alpha=0.7)
    axes[0, 0].plot(initial_state[0], initial_state[1], 'go', markersize=12, label='Start')
    axes[0, 0].set_xlabel('X [m]')
    axes[0, 0].set_ylabel('Y [m]')
    axes[0, 0].set_title('Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Plot 2: Tracking error
    axes[0, 1].plot(time_mpc, mpc_errors * 1000, 'b-', label='MPC', linewidth=2, alpha=0.7)
    axes[0, 1].plot(time_dnn, dnn_errors * 1000, 'r-', label='DNN', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Tracking Error [mm]')
    axes[0, 1].set_title('Tracking Error Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error histogram
    axes[0, 2].hist(mpc_errors * 1000, bins=30, alpha=0.5, label='MPC', color='blue', density=True)
    axes[0, 2].hist(dnn_errors * 1000, bins=30, alpha=0.5, label='DNN', color='red', density=True)
    axes[0, 2].set_xlabel('Tracking Error [mm]')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Error Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Control inputs - speed
    axes[1, 0].plot(time_mpc, mpc_traj[:-1, 0], 'b-', label='MPC s', linewidth=2, alpha=0.7)
    axes[1, 0].plot(time_dnn, dnn_traj[:-1, 0], 'r-', label='DNN s', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Speed s [m/s]')
    axes[1, 0].set_title('Speed Control')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Control inputs - angular velocity
    axes[1, 1].plot(time_mpc, mpc_traj[:-1, 2], 'b-', label='MPC ω', linewidth=2, alpha=0.7)
    axes[1, 1].plot(time_dnn, dnn_traj[:-1, 2], 'r-', label='DNN ω', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Angular Velocity ω [rad/s]')
    axes[1, 1].set_title('Angular Velocity Control')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Path parameter
    axes[1, 2].plot(time_mpc, mpc_traj[:-1, 3], 'b-', label='MPC θ_p', linewidth=2, alpha=0.7)
    axes[1, 2].plot(time_dnn, dnn_traj[:-1, 3], 'r-', label='DNN θ_p', linewidth=2, alpha=0.7)
    axes[1, 2].set_xlabel('Time [s]')
    axes[1, 2].set_ylabel('Path Parameter [rad]')
    axes[1, 2].set_title('Progress Along Track')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()


def main():
    print("="*70)
    print(" MPC vs DNN Comparison")
    print("="*70)
    
    # =========================================
    # Setup Reference Trajectory
    # =========================================
    global t_param, x_ref_array, y_ref_array, theta_ref_array
    total_points = 10000
    t_param, x_ref_array, y_ref_array, theta_ref_array = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    print(f"\n[1/6] Reference trajectory created: {len(x_ref_array)} points")
    
    # =========================================
    # Setup MPC
    # =========================================
    nu = 3
    nx_aug = 4
    N = 70
    ts = 0.03
    q = 100
    qtheta = 10
    r_control = 0.1
    r_v = 0.001
    v_ref = 0.2
    
    print("\n[2/6] Building MPC controller...")
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    mpc_solver = mpc.build_optimizer(build_dir="my_optimizers")
    print("      MPC optimizer ready!")
    
    # =========================================
    # Load DNN Model
    # =========================================
    model_path = "models/Colab Trained Model.pth"
    print(f"\n[3/6] Loading DNN model from {model_path}")
    
    ref_feature_dim = 42
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
    # Setup DNN Controller
    # =========================================
    collector = MPCDataCollector(mpc_solver, t_param, x_ref_array, y_ref_array, 
                                 theta_ref_array, N, nu, ts, v_ref)
    dnn_controller = DNNController(model, collector, t_param, v_ref=v_ref)
    print("\n[4/6] DNN controller ready")
    
    # =========================================
    # Simulate Both Controllers
    # =========================================
    print("\n[5/6] Simulating both controllers...")
    
    # Initial condition: right side pointing up
    initial_state = np.array([0.2, 0.0, np.pi/2])
    idx_closest = find_projection_index(initial_state, x_ref_array, y_ref_array)
    theta_p_initial = t_param[idx_closest]
    aug_initial = np.concatenate([initial_state, [theta_p_initial]])
    
    steps_for_lap = int(2 * np.pi / v_ref / ts) + 100
    
    print(f"  Initial state: x={initial_state[0]:.2f}, y={initial_state[1]:.2f}, θ={initial_state[2]:.2f} rad")
    print(f"  Simulating {steps_for_lap} steps (~{steps_for_lap * ts:.1f} seconds)")
    
    # Simulate MPC
    print("  Simulating MPC...")
    mpc_traj, mpc_controls = simulate_mpc(mpc_solver, aug_initial.copy(), ts, steps_for_lap, 
                                         t_param, x_ref_array, y_ref_array, theta_ref_array)
    mpc_rms, mpc_max, mpc_errors = compute_tracking_error(mpc_traj, x_ref_array, y_ref_array)
    print(f"    MPC: RMS = {mpc_rms*1000:.2f} mm, Max = {mpc_max*1000:.2f} mm")
    
    # Simulate DNN
    print("  Simulating DNN...")
    dnn_traj, dnn_controls = simulate_dnn(dnn_controller, aug_initial.copy(), ts, steps_for_lap)
    dnn_rms, dnn_max, dnn_errors = compute_tracking_error(dnn_traj, x_ref_array, y_ref_array)
    print(f"    DNN: RMS = {dnn_rms*1000:.2f} mm, Max = {dnn_max*1000:.2f} mm")
    
    # =========================================
    # Summary and Visualization
    # =========================================
    print("\n[6/6] Performance Summary:")
    print(f"  {'Metric':<20} {'MPC':<15} {'DNN':<15} {'Difference':<15}")
    print(f"  {'-'*65}")
    print(f"  {'RMS Error [mm]':<20} {mpc_rms*1000:<15.2f} {dnn_rms*1000:<15.2f} {(dnn_rms-mpc_rms)*1000:<15.2f}")
    print(f"  {'Max Error [mm]':<20} {mpc_max*1000:<15.2f} {dnn_max*1000:<15.2f} {(dnn_max-mpc_max)*1000:<15.2f}")
    print(f"  {'Lap Time [s]':<20} {len(mpc_traj)*ts:<15.2f} {len(dnn_traj)*ts:<15.2f} {(len(dnn_traj)-len(mpc_traj))*ts:<15.2f}")
    
    # Plot comparison
    plot_comparison(mpc_traj, dnn_traj, mpc_errors, dnn_errors, x_ref_array, y_ref_array, 
                   initial_state, ts)
    
    print("\n" + "="*70)
    print(" Comparison Complete!")
    print("="*70)
    print(f"\nVisualization saved to: outputs/mpc_dnn_comparison.png")


if __name__ == "__main__":
    main()
