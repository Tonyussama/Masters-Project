"""
DAgger (Dataset Aggregation) training script for MPC to DNN.

This script implements iterative dataset augmentation:
1. Train DNN on initial MPC dataset
2. Run DNN in closed-loop simulation
3. Collect states visited by DNN
4. Label them with MPC expert
5. Add to dataset and retrain
6. Repeat for multiple rounds

Also includes bias and impulse disturbances for robustness.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.mpc_controller import MPCController
from controllers.dnn_controller import MPCNeuralNetwork, DNNController
from utils.reference import generate_ellipse_reference, find_projection_index
from utils.data_collection import MPCDataCollector
from training.trainer import prepare_dataloaders, train_model


def simulate_with_disturbances(controller, initial_state, ts, steps=300, 
                                bias_prob=0.05, impulse_prob=0.02, bias_range=0.1, impulse_range=0.2):
    """
    Simulate closed-loop trajectory with random disturbances.
    
    Args:
        controller: DNNController instance
        initial_state: Initial geometric state [x, y, theta]
        ts: Time step
        steps: Number of simulation steps
        bias_prob: Probability of adding bias disturbance
        impulse_prob: Probability of adding impulse disturbance
        bias_range: Maximum bias magnitude
        impulse_range: Maximum impulse magnitude
    
    Returns:
        trajectory: Array of augmented states
        controls: Array of control actions
        disturbance_log: List of applied disturbances
    """
    trajectory = [initial_state.copy()]
    controls = []
    disturbance_log = []
    
    # Convert initial geometric state to augmented state
    idx_closest = find_projection_index(initial_state, controller.collector.x_ref_array, controller.collector.y_ref_array)
    theta_p_initial = controller.t_param[idx_closest]
    aug_current = np.concatenate([initial_state, [theta_p_initial]])
    
    for step in range(steps):
        u = controller.get_control(aug_current)
        s, w, v_path = u[0], u[1], u[2]
        controls.append(u)
        
        # Apply disturbances
        bias = np.zeros(3)
        impulse = np.zeros(3)
        
        if np.random.random() < bias_prob:
            # Add persistent bias (affects next steps)
            bias = np.random.uniform(-bias_range, bias_range, 3)
            disturbance_log.append({
                'step': step,
                'type': 'bias',
                'magnitude': bias
            })
        
        if np.random.random() < impulse_prob:
            # Add impulse (single-step disturbance)
            impulse = np.random.uniform(-impulse_range, impulse_range, 3)
            disturbance_log.append({
                'step': step,
                'type': 'impulse',
                'magnitude': impulse
            })
        
        # Apply dynamics with disturbances
        aug_current = np.array([
            aug_current[0] + ts * (s + bias[0] + impulse[0]) * np.cos(aug_current[2]),
            aug_current[1] + ts * (s + bias[0] + impulse[0]) * np.sin(aug_current[2]),
            aug_current[2] + ts * (w + bias[1] + impulse[1]),
            aug_current[3] + ts * (v_path + bias[2] + impulse[2])
        ])
        
        # Wrap angle and path parameter
        aug_current[2] = (aug_current[2] + np.pi) % (2 * np.pi) - np.pi
        aug_current[3] = aug_current[3] % (2 * np.pi)
        trajectory.append(aug_current.copy())
    
    return np.array(trajectory), np.array(controls), disturbance_log


def collect_on_policy_data(controller, mpc_solver, collector, num_trajectories=10, steps_per_traj=300,
                           bias_prob=0.05, impulse_prob=0.02):
    """
    Collect on-policy data by running DNN in closed-loop and labeling with MPC.
    
    Args:
        controller: DNNController instance
        mpc_solver: MPC solver for labeling
        collector: MPCDataCollector instance
        num_trajectories: Number of trajectories to simulate
        steps_per_traj: Steps per trajectory
        bias_prob: Probability of bias disturbance
        impulse_prob: Probability of impulse disturbance
    
    Returns:
        new_states: New states collected
        new_references: New reference features
        new_controls: MPC-labeled controls
    """
    new_states = []
    new_references = []
    new_controls = []
    
    print(f"\nCollecting on-policy data from {num_trajectories} trajectories...")
    
    for traj_idx in tqdm(range(num_trajectories), desc="Trajectories"):
        # Random initial condition near the reference path
        idx = np.random.randint(0, len(collector.x_ref_array))
        x0 = collector.x_ref_array[idx] + np.random.normal(0, 0.15)
        y0 = collector.y_ref_array[idx] + np.random.normal(0, 0.15)
        theta0 = collector.theta_ref_array[idx] + np.random.normal(0, 0.3)
        initial_state = np.array([x0, y0, theta0])
        
        # Simulate with disturbances
        trajectory, _, disturbances = simulate_with_disturbances(
            controller, initial_state, collector.ts, steps=steps_per_traj,
            bias_prob=bias_prob, impulse_prob=impulse_prob
        )
        
        if len(disturbances) > 0:
            print(f"  Trajectory {traj_idx}: {len(disturbances)} disturbances applied")
        
        # Label each state with MPC
        for aug_state in trajectory[::5]:  # Subsample every 5th state
            x_ref, y_ref, theta_ref = collector.get_reference_horizon(aug_state[3])
            
            # Solve MPC
            p_vector = np.concatenate([aug_state, x_ref, y_ref, theta_ref])
            u_init = [0.01] * (collector.nu * collector.N)
            
            result = mpc_solver.run(p=p_vector, initial_guess=u_init)
            
            if result.exit_status == 'Converged':
                u_optimal = result.solution[0:collector.nu]
                
                new_states.append(aug_state[:3])
                ref_features = collector.extract_reference_features(
                    aug_state[:3], x_ref, y_ref, theta_ref
                )
                new_references.append(ref_features)
                new_controls.append(u_optimal)
    
    print(f"Collected {len(new_states)} new on-policy samples")
    return np.array(new_states), np.array(new_references), np.array(new_controls)


def main():
    print("="*70)
    print(" DAgger Training: Iterative Dataset Aggregation")
    print("="*70)
    
    # =========================================
    # Setup
    # =========================================
    total_points = 10000
    t_param, x_ref_array, y_ref_array, theta_ref_array = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    print(f"\nReference trajectory created: {len(x_ref_array)} points")
    
    nu = 3
    nx_aug = 4
    N = 70
    ts = 0.03
    q = 100
    qtheta = 10
    r_control = 0.1
    r_v = 0.001
    v_ref = 0.2
    
    print("Building MPC optimizer...")
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    solver = mpc.build_optimizer(build_dir="my_optimizers")
    print("MPC optimizer ready!")
    
    collector = MPCDataCollector(solver, t_param, x_ref_array, y_ref_array, 
                                 theta_ref_array, N, nu, ts, v_ref)
    
    # =========================================
    # Round 0: Initial Data Collection
    # =========================================
    print("\n" + "="*70)
    print(" Round 0: Initial MPC Dataset")
    print("="*70)
    
    states, references, controls, failure_info, block_ids = collector.collect_data_corridor(
        num_anchors=5000, offset_distance=0.1, 
        heading_error_range=np.pi/6, keep_failures=True
    )
    
    print(f"Initial dataset: {len(states)} samples")
    
    # =========================================
    # DAgger Iterations
    # =========================================
    n_dagger_rounds = 3
    trajectories_per_round = 10
    
    for round_idx in range(n_dagger_rounds):
        print("\n" + "="*70)
        print(f" DAgger Round {round_idx + 1}/{n_dagger_rounds}")
        print("="*70)
        
        # Prepare data loaders
        train_loader, val_loader, dataset = prepare_dataloaders(
            references, controls, train_split=0.8, batch_size=32,
            block_ids=block_ids, standardize_targets=True
        )
        
        # Train model
        ref_feature_dim = references.shape[1]
        model = MPCNeuralNetwork(
            ref_feature_dim=ref_feature_dim, 
            hidden_dims=[1024, 1024, 512, 256, 128], 
            output_dim=2
        )
        
        trained_model = train_model(
            model, train_loader, val_loader, references, 
            epochs=1000, lr=0.001, seed=42 + round_idx
        )
        
        # Create DNN controller
        dnn_controller = DNNController(trained_model, collector, t_param, v_ref=v_ref)
        
        # Collect on-policy data
        new_states, new_references, new_controls = collect_on_policy_data(
            dnn_controller, solver, collector, 
            num_trajectories=trajectories_per_round, 
            steps_per_traj=300,
            bias_prob=0.05, impulse_prob=0.02
        )
        
        if len(new_states) > 0:
            # Augment dataset
            states = np.concatenate([states, new_states])
            references = np.concatenate([references, new_references])
            controls = np.concatenate([controls, new_controls])
            
            # Update block IDs (assign new samples to new blocks)
            new_block_ids = np.arange(len(block_ids), len(block_ids) + len(new_states))
            block_ids = np.concatenate([block_ids, new_block_ids])
            
            print(f"Dataset after round {round_idx + 1}: {len(states)} total samples")
        else:
            print("No new samples collected - skipping to next round")
    
    # =========================================
    # Final Training
    # =========================================
    print("\n" + "="*70)
    print(" Final Training on Augmented Dataset")
    print("="*70)
    
    train_loader, val_loader, dataset = prepare_dataloaders(
        references, controls, train_split=0.8, batch_size=32,
        block_ids=block_ids, standardize_targets=True
    )
    
    model = MPCNeuralNetwork(
        ref_feature_dim=references.shape[1], 
        hidden_dims=[1024, 1024, 512, 256, 128], 
        output_dim=2
    )
    
    final_model = train_model(
        model, train_loader, val_loader, references, 
        epochs=1000, lr=0.001, seed=999
    )
    
    # Save final model
    torch.save(final_model.state_dict(), 'models/dagger_mpc_dnn_model.pth')
    print("\nFinal model saved as 'models/dagger_mpc_dnn_model.pth'")
    
    # Save augmented dataset
    np.savez('data/dagger_mpc_dataset.npz',
             states=states,
             references=references,
             controls=controls,
             block_ids=block_ids,
             x_ref_array=x_ref_array,
             y_ref_array=y_ref_array,
             t_param=t_param,
             theta_ref_array=theta_ref_array,
             N=N, nu=nu, ts=ts)
    print("Augmented dataset saved to 'data/dagger_mpc_dataset.npz'")
    
    print("\n" + "="*70)
    print(" DAgger Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
