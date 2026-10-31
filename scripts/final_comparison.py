"""
Final comparison of MPC, LQR, and DNN controllers on the ellipse.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.mpc_controller import MPCController
from controllers.dnn_controller import MPCNeuralNetwork
from utils.reference import generate_ellipse_reference, find_projection_index


def final_comparison():
    """Compare MPC and DNN on the ellipse."""
    print("="*70)
    print(" Final Controller Comparison: MPC vs DNN")
    print("="*70)
    
    # Generate reference trajectory
    total_points = 10000
    t_param, x_ref, y_ref, theta_ref = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    
    # Calculate curvature
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    
    dt = 0.03
    v_ref = 0.2
    max_steps = 3000
    
    # Initial state - start on the path with small offset
    initial_geometric_state = np.array([x_ref[0] + 0.01, y_ref[0] + 0.01, theta_ref[0] + 0.01])
    idx_closest = find_projection_index(initial_geometric_state, x_ref, y_ref)
    theta_p_initial = t_param[idx_closest]
    initial_aug_state = np.concatenate([initial_geometric_state, [theta_p_initial]])
    
    # =========================================
    # MPC Simulation
    # =========================================
    print(f"\n[1/2] Simulating MPC...")
    mpc = MPCController(nu=3, nx_aug=4, N=70, ts=dt, q=100, qtheta=10,
                       r_control=0.1, r_v=0.001, v_ref=v_ref)
    mpc.build_optimizer(build_dir="my_optimizers")

    state = initial_aug_state.copy()
    traj_mpc = [state.copy()]
    ctrl_mpc = []
    err_mpc = []

    start_pos = state[:2]
    lap_mpc = False

    for step in range(max_steps):
        theta_p_current = state[3]
        N = 70
        theta_ref_horizon_p = theta_p_current + v_ref * np.arange(N) * dt

        x_ref_horizon = np.interp(theta_ref_horizon_p, t_param, x_ref)
        y_ref_horizon = np.interp(theta_ref_horizon_p, t_param, y_ref)
        theta_ref_unwrapped = np.unwrap(theta_ref)
        theta_ref_horizon = np.interp(theta_ref_horizon_p, t_param, theta_ref_unwrapped)
        theta_ref_horizon = (theta_ref_horizon + np.pi) % (2 * np.pi) - np.pi

        u = mpc.get_first_control(state, x_ref_horizon, y_ref_horizon, theta_ref_horizon)
        ctrl_mpc.append(u)

        idx = find_projection_index(state, x_ref, y_ref)
        error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
        err_mpc.append(error)

        s, omega, v_path = u[0], u[1], u[2]
        state = np.array([
            state[0] + dt * s * np.cos(state[2]),
            state[1] + dt * s * np.sin(state[2]),
            state[2] + dt * omega,
            state[3] + dt * v_path
        ])
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        state[3] = state[3] % (2 * np.pi)

        traj_mpc.append(state.copy())

        # Detect lap completion by actual position (distance to start)
        if step > 1000:
            dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
            if dist_to_start < 0.1:
                lap_mpc = True
                break

    traj_mpc = np.array(traj_mpc)
    ctrl_mpc = np.array(ctrl_mpc)
    err_mpc = np.array(err_mpc)
    print(f"      MPC: Mean error={np.mean(err_mpc)*1000:.2f}mm, Lap={lap_mpc}")

    # =========================================
    # DNN Simulation (using pre-trained model)
    # =========================================
    print(f"\n[2/2] Simulating DNN...")
    import torch
    import torch.nn as nn
    import os

    # Check if model exists
    model_path = 'models/final_mpc_dnn_model.pth'
    if not os.path.exists(model_path):
        print(f"      Warning: Model file not found at '{model_path}'")
        print(f"      Skipping DNN simulation. Train the model first using train_mpc_to_dnn.py or colab_training.ipynb")
        traj_dnn = np.array([[0, 0, 0, 0]])
        ctrl_dnn = np.array([[0, 0, 0]])
        err_dnn = np.array([0])
        lap_dnn = False
    else:
        # Use the same model architecture as training
        model = MPCNeuralNetwork(ref_feature_dim=42, hidden_dims=[1024, 1024, 512, 256, 128], output_dim=2)
        model.load_state_dict(torch.load('models/final_mpc_dnn_model.pth', weights_only=True))
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Load dataset for normalization
        data = np.load('data/mpc_dataset.npz')
        references = data['references']
        model.fit_normalization(references)

        # Simulate
        state = initial_geometric_state.copy()
        traj_dnn = [np.concatenate([state, [theta_p_initial]]).copy()]
        ctrl_dnn = []
        err_dnn = []

        start_pos = state[:2]
        lap_dnn = False

        # Track path parameter for DNN
        theta_p_dnn = theta_p_initial

        with torch.no_grad():
            for step in range(max_steps):
                idx = find_projection_index(state, x_ref, y_ref)
                theta_p_current = t_param[idx]
                theta_ref_horizon_p = theta_p_current + v_ref * np.arange(70) * dt

                x_ref_horizon = np.interp(theta_ref_horizon_p, t_param, x_ref)
                y_ref_horizon = np.interp(theta_ref_horizon_p, t_param, y_ref)
                theta_ref_unwrapped = np.unwrap(theta_ref)
                theta_ref_horizon = np.interp(theta_ref_horizon_p, t_param, theta_ref_unwrapped)
                theta_ref_horizon = (theta_ref_horizon + np.pi) % (2 * np.pi) - np.pi

                # Extract reference features (robot-frame)
                x, y, theta_state = state
                cos_theta = np.cos(-theta_state)
                sin_theta = np.sin(-theta_state)

                dx = x_ref_horizon - x
                dy = y_ref_horizon - y

                x_ref_local = dx * cos_theta - dy * sin_theta
                y_ref_local = dx * sin_theta + dy * cos_theta
                theta_ref_local = theta_ref_horizon - theta_state
                theta_ref_local = np.arctan2(np.sin(theta_ref_local), np.cos(theta_ref_local))

                stride = 5
                ref_features = np.concatenate([
                    x_ref_local[::stride],
                    y_ref_local[::stride],
                    theta_ref_local[::stride]
                ])

                ref_tensor = torch.FloatTensor(ref_features).unsqueeze(0).to(device)
                u = model(ref_tensor).cpu().numpy()[0]
                s_omega = u  # Returns [s, omega]
                ctrl_dnn.append(np.array([s_omega[0], s_omega[1], v_ref]))

                error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
                err_dnn.append(error)

                s, omega = s_omega[0], s_omega[1]
                state = np.array([
                    state[0] + dt * s * np.cos(state[2]),
                    state[1] + dt * s * np.sin(state[2]),
                    state[2] + dt * omega
                ])
                state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi

                # Update path parameter
                theta_p_dnn = (theta_p_dnn + dt * v_ref) % (2 * np.pi)
                traj_dnn.append(np.concatenate([state, [theta_p_dnn]]).copy())

                # Detect lap completion by actual position (distance to start)
                if step > 1000:
                    dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
                    if dist_to_start < 0.1:
                        lap_dnn = True
                        break

        traj_dnn = np.array(traj_dnn)
        ctrl_dnn = np.array(ctrl_dnn)
        err_dnn = np.array(err_dnn)
        print(f"      DNN: Mean error={np.mean(err_dnn)*1000:.2f}mm, Lap={lap_dnn}")
    
    # =========================================
    # Visualization
    # =========================================
    print(f"\n[3/3] Generating visualization...")

    fig = plt.figure(figsize=(20, 14))

    # Plot 1: All trajectories
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Reference')
    ax1.plot(traj_mpc[:, 0], traj_mpc[:, 1], 'b-', linewidth=2, label='MPC', alpha=0.8)
    if len(traj_dnn) > 1:
        ax1.plot(traj_dnn[:, 0], traj_dnn[:, 1], 'r-', linewidth=2, label='DNN', alpha=0.8)
    ax1.plot(initial_aug_state[0], initial_aug_state[1], 'ko', markersize=10, label='Start')
    ax1.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax1.set_title('All Trajectories on Ellipse', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    #ax1.axis('equal')

    # Plot 2: Tracking errors
    ax2 = plt.subplot(3, 3, 2)
    times_mpc = np.arange(len(err_mpc)) * dt
    times_dnn = np.arange(len(err_dnn)) * dt if len(traj_dnn) > 1 else []
    ax2.plot(times_mpc, err_mpc * 1000, 'b-', linewidth=2, label='MPC')
    if len(traj_dnn) > 1:
        ax2.plot(times_dnn, err_dnn * 1000, 'r-', linewidth=2, label='DNN')
    ax2.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax2.set_title('Tracking Error Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Linear velocity
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(times_mpc, ctrl_mpc[:, 0], 'b-', linewidth=2, label='MPC')
    if len(traj_dnn) > 1:
        ax3.plot(times_dnn, ctrl_dnn[:, 0], 'r-', linewidth=2, label='DNN')
    ax3.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax3.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Linear Velocity [m/s]', fontsize=11, fontweight='bold')
    ax3.set_title('Linear Velocity Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Angular velocity
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(times_mpc, ctrl_mpc[:, 1], 'b-', linewidth=2, label='MPC')
    if len(traj_dnn) > 1:
        ax4.plot(times_dnn, ctrl_dnn[:, 1], 'r-', linewidth=2, label='DNN')
    ax4.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Angular Velocity [rad/s]', fontsize=11, fontweight='bold')
    ax4.set_title('Angular Velocity Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Error distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(err_mpc * 1000, bins=30, alpha=0.5, label='MPC', color='blue')
    if len(traj_dnn) > 1:
        ax5.hist(err_dnn * 1000, bins=30, alpha=0.5, label='DNN', color='red')
    ax5.set_xlabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Performance bar chart
    ax6 = plt.subplot(3, 3, 6)
    controllers = ['MPC', 'DNN']
    mean_errors = [np.mean(err_mpc)*1000, np.mean(err_dnn)*1000 if len(traj_dnn) > 1 else 0]
    lap_status = [lap_mpc, lap_dnn]
    colors = ['blue', 'red']
    bars = ax6.bar(controllers, mean_errors, color=colors, alpha=0.7)
    ax6.set_ylabel('Mean Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax6.set_title('Mean Error Comparison', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add lap status labels
    for i, (bar, lap) in enumerate(zip(bars, lap_status)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{"✓" if lap else "✗"}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Plot 7: Cross-track error along path
    ax7 = plt.subplot(3, 3, 7)
    path_params_mpc = traj_mpc[:-1, 3]  # Exclude last point to match error array
    path_params_dnn = traj_dnn[:-1, 3] if len(traj_dnn) > 1 else []
    ax7.plot(path_params_mpc, err_mpc, 'b-', linewidth=2, label='MPC', alpha=0.8)
    if len(traj_dnn) > 1:
        ax7.plot(path_params_dnn, err_dnn, 'r-', linewidth=2, label='DNN', alpha=0.8)
    ax7.set_xlabel('Path Parameter [rad]', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Cross-track Error [m]', fontsize=11, fontweight='bold')
    ax7.set_title('Cross-track Error Along Path', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    # Plot 8: Path parameter velocity (MPC and DNN only)
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(times_mpc, ctrl_mpc[:, 2], 'b-', linewidth=2, label='MPC', alpha=0.8)
    if len(traj_dnn) > 1:
        ax8.plot(times_dnn, ctrl_dnn[:, 2], 'r-', linewidth=2, label='DNN', alpha=0.8)
    ax8.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax8.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Path Velocity [rad/s]', fontsize=11, fontweight='bold')
    ax8.set_title('Path Parameter Velocity', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim([0, 0.5])

    # Plot 9: Lap completion status
    ax9 = plt.subplot(3, 3, 9)
    controllers = ['MPC', 'DNN']
    lap_status = [lap_mpc, lap_dnn]
    colors = ['blue', 'red']
    bars = ax9.bar(controllers, [1 if lap else 0 for lap in lap_status], color=colors, alpha=0.7)
    ax9.set_ylabel('Lap Completed', fontsize=11, fontweight='bold')
    ax9.set_title('Lap Completion Status', fontsize=12, fontweight='bold')
    ax9.set_ylim([0, 1.2])
    ax9.set_yticks([0, 1])
    ax9.set_yticklabels(['No', 'Yes'])
    ax9.grid(True, alpha=0.3, axis='y')

    # Add lap status labels
    for bar, lap in zip(bars, lap_status):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{"✓" if lap else "✗"}',
                ha='center', va='bottom', fontsize=20, fontweight='bold')

    plt.suptitle('Final Controller Comparison: MPC vs DNN', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/final_controller_comparison.png', dpi=200, bbox_inches='tight')

    print(f"      Visualization saved to 'outputs/final_controller_comparison.png'")
    
    # Summary table
    print("\n" + "="*70)
    print(" Final Summary")
    print("="*70)
    print(f"{'Controller':<12} {'Mean Error':<15} {'Max Error':<15} {'Lap':<10}")
    print("-" * 70)
    print(f"{'MPC':<12} {np.mean(err_mpc)*1000:<15.2f} {np.max(err_mpc)*1000:<15.2f} {str(lap_mpc):<10}")
    if len(traj_dnn) > 1:
        print(f"{'DNN':<12} {np.mean(err_dnn)*1000:<15.2f} {np.max(err_dnn)*1000:<15.2f} {str(lap_dnn):<10}")
    else:
        print(f"{'DNN':<12} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
    print("="*70)
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    final_comparison()
