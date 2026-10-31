"""
Main script for MPC to DNN training pipeline.
This script:
1. Sets up the MPC controller
2. Collects training data from MPC
3. Trains a neural network to mimic MPC
4. Evaluates the trained DNN controller
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.mpc_controller import MPCController
from controllers.dnn_controller import MPCNeuralNetwork, DNNController
from utils.reference import generate_ellipse_reference, find_projection_index
from utils.data_collection import MPCDataCollector
from utils.visualization import plot_comprehensive_results
from training.trainer import prepare_dataloaders, train_model


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
    """
    trajectory = [initial_state.copy()]
    controls = []
    
    # Convert initial geometric state to augmented state
    idx_closest = find_projection_index(initial_state, controller.collector.x_ref_array, controller.collector.y_ref_array)
    theta_p_initial = controller.t_param[idx_closest]
    aug_current = np.concatenate([initial_state, [theta_p_initial]])
    
    for _ in range(steps):
        u = controller.get_control(aug_current)
        s, w, v_path = u[0], u[1], u[2]
        controls.append(u)
        
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
        trajectory.append(aug_current.copy())
    
    return np.array(trajectory), np.array(controls)


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
    
    return rms_error, max_error


def main():
    print("="*70)
    print(" MPC to DNN: Complete Training Pipeline")
    print("="*70)
    
    # =========================================
    # PART 1: Setup Reference Trajectory
    # =========================================
    total_points = 10000
    t_param, x_ref_array, y_ref_array, theta_ref_array = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    print(f"\n[1/7] Reference trajectory created: {len(x_ref_array)} points")
    
    # =========================================
    # PART 2: Build MPC Controller
    # =========================================
    # Define MPC parameters
    nu = 3  # [s, omega, v]
    nx_aug = 4  # [x, y, theta, theta_p]
    N = 70
    ts = 0.03
    q = 100
    qtheta = 10
    r_control = 0.1
    r_v = 0.001
    v_ref = 0.2
    
    print("[2/7] Building MPC optimizer...")
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    solver = mpc.build_optimizer(build_dir="my_optimizers")
    print("      MPC optimizer ready!")
    
    # =========================================
    # PART 3: Collect Training Data
    # =========================================
    collector = MPCDataCollector(solver, t_param, x_ref_array, y_ref_array, 
                                 theta_ref_array, N, nu, ts, v_ref)
    states, references, controls, failure_info, block_ids = collector.collect_data_corridor(
        num_anchors=10000, offset_distance=0.1,
        heading_error_range=np.pi/6, keep_failures=True
    )
    
    # Save dataset
    np.savez('data/mpc_dataset.npz',
             states=states,
             references=references,
             controls=controls,
             block_ids=block_ids,
             x_ref_array=x_ref_array,
             y_ref_array=y_ref_array,
             t_param=t_param,
             theta_ref_array=theta_ref_array,
             N=N, nu=nu, ts=ts)
    print("      Dataset saved to 'data/mpc_dataset.npz'")
    
    # =========================================
    # PART 4: Prepare Training Data
    # =========================================
    print("\n[5/7] Preparing training data...")
    # Use larger batch size for GPU (256), smaller for CPU (32)
    batch_size = 256 if torch.cuda.is_available() else 32
    print(f"      Using batch_size={batch_size} ({'GPU' if torch.cuda.is_available() else 'CPU'})")
    
    train_loader, val_loader, dataset = prepare_dataloaders(
        references, controls, train_split=0.8, batch_size=batch_size,
        block_ids=block_ids, standardize_targets=True
    )
    print(f"      Training: {len(train_loader.dataset)} samples | Validation: {len(val_loader.dataset)} samples")
    
    # =========================================
    # PART 5: Train Neural Network
    # =========================================
    ref_feature_dim = references.shape[1]
    print(f"      Network input: references({ref_feature_dim}) dims (robot-frame only)")
    
    model = MPCNeuralNetwork(
        ref_feature_dim=ref_feature_dim,
        hidden_dims=[1024, 1024, 512, 256, 128],
        output_dim=2
    )
    
    trained_model = train_model(model, train_loader, val_loader, references, epochs=1000, lr=0.001, seed=42, batch_size=batch_size)
    
    # Save model
    torch.save(trained_model.state_dict(), 'models/final_mpc_dnn_model.pth')
    print("\n[6/7] Model saved as 'models/final_mpc_dnn_model.pth'")
    
    # =========================================
    # PART 6: Test DNN Controller
    # =========================================
    print("[7/7] Testing DNN controller on multiple initial conditions...")
    dnn_controller = DNNController(trained_model, collector, t_param, v_ref=v_ref)
    
    test_initial_states = [
        np.array([0.3, 0.0, np.pi/3]),
        np.array([0.2, 0.5, 0.0]),
        np.array([-0.15, -0.3, -np.pi/4]),
        np.array([0.0, 1.0, np.pi/2]),
        np.array([0.15, -0.4, -np.pi/6]),
    ]
    
    trajectories = []
    controls_list = []
    tracking_errors = []
    
    for i, init_state in enumerate(test_initial_states):
        traj, controls = simulate_trajectory(dnn_controller, init_state, ts, steps=300)
        trajectories.append(traj)
        controls_list.append(controls)
        
        rms_error, max_error = compute_tracking_error(traj, x_ref_array, y_ref_array, t_param)
        tracking_errors.append({'rms': rms_error, 'max': max_error})
        print(f"  Test {i+1}: RMS = {rms_error*1000:.2f} mm, Max = {max_error*1000:.2f} mm")
    
    # Print summary statistics
    rms_errors = [e['rms'] for e in tracking_errors]
    max_errors = [e['max'] for e in tracking_errors]
    print(f"\n  Average RMS error: {np.mean(rms_errors)*1000:.2f} mm")
    print(f"  Average Max error: {np.mean(max_errors)*1000:.2f} mm")
    
    plot_comprehensive_results(trajectories, controls_list, x_ref_array, y_ref_array, test_initial_states, ts)
    
    print("\n" + "="*70)
    print(" ✓ Training and Evaluation Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  • data/mpc_dataset.npz                 (Training data)")
    print("  • models/best_mpc_dnn_model.pth        (Best model during training)")
    print("  • models/final_mpc_dnn_model.pth       (Final model)")
    print("  • outputs/training_curves.png          (Training progress)")
    print("  • outputs/dnn_results_comprehensive.png (Performance analysis)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
