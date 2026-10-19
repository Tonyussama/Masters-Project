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
    states, references, controls = collector.collect_data_corridor(num_anchors=1000, offset_distance=0.1)
    
    # Save dataset
    np.savez('data/mpc_dataset.npz',
             states=states,
             references=references,
             controls=controls,
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
    train_loader, val_loader, dataset = prepare_dataloaders(
        states, references, controls, train_split=0.8, batch_size=32
    )
    print(f"      Training: {len(train_loader.dataset)} samples | Validation: {len(val_loader.dataset)} samples")
    
    # =========================================
    # PART 5: Train Neural Network
    # =========================================
    ref_feature_dim = references.shape[1]
    print(f"      Network input: state(3) + references({ref_feature_dim}) = {3 + ref_feature_dim} dims")
    
    model = MPCNeuralNetwork(state_dim=3, ref_feature_dim=ref_feature_dim, 
                            hidden_dims=[128, 128, 64, 32], output_dim=3)
    
    trained_model = train_model(model, train_loader, val_loader, states, references, epochs=300, lr=0.001)
    
    # Save model
    torch.save(trained_model.state_dict(), 'models/final_mpc_dnn_model.pth')
    print("\n[6/7] Model saved as 'models/final_mpc_dnn_model.pth'")
    
    # =========================================
    # PART 6: Test DNN Controller
    # =========================================
    print("[7/7] Testing DNN controller on multiple initial conditions...")
    dnn_controller = DNNController(trained_model, collector, t_param)
    
    test_initial_states = [
        np.array([0.3, 0.0, np.pi/3]),
        np.array([0.2, 0.5, 0.0]),
        np.array([-0.15, -0.3, -np.pi/4]),
        np.array([0.0, 1.0, np.pi/2]),
        np.array([0.15, -0.4, -np.pi/6]),
    ]
    
    trajectories = []
    controls_list = []
    
    for i, init_state in enumerate(test_initial_states):
        traj, controls = simulate_trajectory(dnn_controller, init_state, ts, steps=300)
        trajectories.append(traj)
        controls_list.append(controls)
    
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
