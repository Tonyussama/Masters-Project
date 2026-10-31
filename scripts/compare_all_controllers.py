"""
Compare all four controllers: Stanley, LQR, MPC, and DNN.

This script:
1. Simulates all controllers from the same initial condition
2. Creates comprehensive visualizations
3. Computes comparative performance metrics
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.stanley_controller import StanleyController
from controllers.lqr_controller import LQRController
from controllers.mpc_controller import MPCController
from controllers.dnn_controller import MPCNeuralNetwork, DNNController
from utils.reference import generate_ellipse_reference, find_projection_index
from utils.data_collection import MPCDataCollector


def simulate_lap(controller, aug_state, ts, t_param, x_ref_array, y_ref_array, theta_ref_array, controller_type):
    """Simulate controller trajectory until lap completion."""
    trajectory = [aug_state.copy()]
    controls = []
    
    initial_theta_p = aug_state[3]
    
    while True:
        if controller_type == 'mpc':
            # MPC needs reference horizon
            theta_p = aug_state[3]
            x_ref = np.interp(theta_p + np.arange(70) * ts * 0.2, t_param, x_ref_array)
            y_ref = np.interp(theta_p + np.arange(70) * ts * 0.2, t_param, y_ref_array)
            theta_ref = np.interp(theta_p + np.arange(70) * ts * 0.2, t_param, np.unwrap(theta_ref_array))
            theta_ref = (theta_ref + np.pi) % (2 * np.pi) - np.pi
            
            p_vector = np.concatenate([aug_state, x_ref, y_ref, theta_ref])
            u_init = [0.01] * (3 * 70)
            result = controller.run(p=p_vector, initial_guess=u_init)
            
            if result.exit_status == 'Converged':
                u = result.solution[0:3]
            else:
                u = controls[-1] if controls else np.array([0.01, 0.0, 0.2])
        else:
            # Stanley, LQR, DNN - now all use aug_state
            u = controller.get_control(aug_state)
        
        controls.append(u)
        
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
        trajectory.append(aug_current.copy())
        
        # Check if lap complete
        if aug_state[3] - initial_theta_p >= 2 * np.pi:
            break
        
        if len(trajectory) > 2000:
            break
    
    return np.array(trajectory), np.array(controls)


def compute_tracking_error(trajectory, x_ref_array, y_ref_array):
    """Compute tracking error."""
    errors = []
    for aug_state in trajectory[:-1]:
        idx = find_projection_index(aug_state[:2], x_ref_array, y_ref_array)
        x_path = x_ref_array[idx]
        y_path = y_ref_array[idx]
        error = np.sqrt((aug_state[0] - x_path)**2 + (aug_state[1] - y_path)**2)
        errors.append(error)
    errors = np.array(errors)
    rms_error = np.sqrt(np.mean(errors**2))
    max_error = np.max(errors)
    return rms_error, max_error, errors


def plot_all_controllers(results, x_ref, y_ref, initial_state, ts, save_path='outputs/all_controllers_comparison.png'):
    """Plot comparison of all four controllers."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = {'Stanley': 'green', 'LQR': 'orange', 'MPC': 'blue', 'DNN': 'red'}
    
    # Plot 1: Trajectories
    axes[0, 0].plot(x_ref, y_ref, 'k--', label='Reference', linewidth=2, alpha=0.5)
    for name, result in results.items():
        traj = result['trajectory']
        axes[0, 0].plot(traj[:, 0], traj[:, 1], color=colors[name], label=name, linewidth=2, alpha=0.7)
    axes[0, 0].plot(initial_state[0], initial_state[1], 'go', markersize=12, label='Start')
    axes[0, 0].set_xlabel('X [m]')
    axes[0, 0].set_ylabel('Y [m]')
    axes[0, 0].set_title('Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Tracking error
    for name, result in results.items():
        errors = result['errors']
        time = np.arange(len(errors)) * ts
        axes[0, 1].plot(time, errors * 1000, color=colors[name], label=name, linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Tracking Error [mm]')
    axes[0, 1].set_title('Tracking Error Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error histogram
    for name, result in results.items():
        errors = result['errors']
        axes[0, 2].hist(errors * 1000, bins=30, alpha=0.5, label=name, color=colors[name], density=True)
    axes[0, 2].set_xlabel('Tracking Error [mm]')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Error Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Speed control
    for name, result in results.items():
        ctrl = result['controls']
        time = np.arange(len(ctrl)) * ts
        axes[1, 0].plot(time, ctrl[:, 0], color=colors[name], label=name, linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Speed s [m/s]')
    axes[1, 0].set_title('Speed Control')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Angular velocity control
    for name, result in results.items():
        ctrl = result['controls']
        time = np.arange(len(ctrl)) * ts
        axes[1, 1].plot(time, ctrl[:, 1], color=colors[name], label=name, linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Angular Velocity ω [rad/s]')
    axes[1, 1].set_title('Angular Velocity Control')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Path parameter
    for name, result in results.items():
        traj = result['trajectory']
        time = np.arange(len(traj) - 1) * ts
        axes[1, 2].plot(time, traj[:-1, 3], color=colors[name], label=name, linewidth=2, alpha=0.7)
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
    print(" All Controllers Comparison: Stanley, LQR, MPC, DNN")
    print("="*70)
    
    # Setup
    total_points = 10000
    t_param, x_ref_array, y_ref_array, theta_ref_array = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    print(f"\n[1/8] Reference trajectory created: {len(x_ref_array)} points")
    
    nu = 3
    nx_aug = 4
    N = 70
    ts = 0.03
    q = 100
    qtheta = 10
    r_control = 0.1
    r_v = 0.001
    v_ref = 0.2
    
    # Stanley
    print("\n[2/8] Setting up Stanley controller...")
    stanley = StanleyController(v_ref=v_ref, dt=ts, k_gain=5.0, k_soft=0.1, k_heading=2.0)
    stanley.set_reference(x_ref_array, y_ref_array, theta_ref_array, t_param)
    
    # LQR
    print("[3/8] Setting up LQR controller...")
    lqr = LQRController(v_ref=v_ref, dt=ts)
    lqr.set_reference(x_ref_array, y_ref_array, theta_ref_array, t_param)
    
    # MPC
    print("[4/8] Building MPC controller...")
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    mpc_solver = mpc.build_optimizer(build_dir="my_optimizers")
    
    # DNN
    print("[5/8] Loading DNN model...")
    model_path = "models/Colab Trained Model.pth"
    ref_feature_dim = 42
    model = MPCNeuralNetwork(
        ref_feature_dim=ref_feature_dim,
        hidden_dims=[1024, 1024, 512, 256, 128],
        output_dim=2
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    collector = MPCDataCollector(mpc_solver, t_param, x_ref_array, y_ref_array, 
                                 theta_ref_array, N, nu, ts, v_ref)
    dnn_controller = DNNController(model, collector, t_param, v_ref=v_ref)
    
    # Simulate all controllers
    print("\n[6/8] Simulating all controllers from on-path start...")
    initial_state = np.array([0.1, 0.0, np.pi/2])
    idx_closest = find_projection_index(initial_state, x_ref_array, y_ref_array)
    theta_p_initial = t_param[idx_closest]
    aug_initial = np.concatenate([initial_state, [theta_p_initial]])
    
    print(f"  Initial state: x={initial_state[0]:.2f}, y={initial_state[1]:.2f}, θ={initial_state[2]:.2f} rad")
    
    results = {}
    
    print("  Simulating Stanley...")
    traj, ctrl = simulate_lap(stanley, aug_initial.copy(), ts, t_param, x_ref_array, y_ref_array, theta_ref_array, 'stanley')
    rms, max_err, errors = compute_tracking_error(traj, x_ref_array, y_ref_array)
    results['Stanley'] = {'trajectory': traj, 'controls': ctrl, 'errors': errors, 'rms': rms, 'max': max_err}
    print(f"    Stanley: RMS = {rms*1000:.2f} mm, Max = {max_err*1000:.2f} mm, Lap time = {len(traj)*ts:.2f} s")
    
    print("  Simulating LQR...")
    traj, ctrl = simulate_lap(lqr, aug_initial.copy(), ts, t_param, x_ref_array, y_ref_array, theta_ref_array, 'lqr')
    rms, max_err, errors = compute_tracking_error(traj, x_ref_array, y_ref_array)
    results['LQR'] = {'trajectory': traj, 'controls': ctrl, 'errors': errors, 'rms': rms, 'max': max_err}
    print(f"    LQR: RMS = {rms*1000:.2f} mm, Max = {max_err*1000:.2f} mm, Lap time = {len(traj)*ts:.2f} s")
    
    print("  Simulating MPC...")
    traj, ctrl = simulate_lap(mpc_solver, aug_initial.copy(), ts, t_param, x_ref_array, y_ref_array, theta_ref_array, 'mpc')
    rms, max_err, errors = compute_tracking_error(traj, x_ref_array, y_ref_array)
    results['MPC'] = {'trajectory': traj, 'controls': ctrl, 'errors': errors, 'rms': rms, 'max': max_err}
    print(f"    MPC: RMS = {rms*1000:.2f} mm, Max = {max_err*1000:.2f} mm, Lap time = {len(traj)*ts:.2f} s")
    
    print("  Simulating DNN...")
    traj, ctrl = simulate_lap(dnn_controller, aug_initial.copy(), ts, t_param, x_ref_array, y_ref_array, theta_ref_array, 'dnn')
    rms, max_err, errors = compute_tracking_error(traj, x_ref_array, y_ref_array)
    results['DNN'] = {'trajectory': traj, 'controls': ctrl, 'errors': errors, 'rms': rms, 'max': max_err}
    print(f"    DNN: RMS = {rms*1000:.2f} mm, Max = {max_err*1000:.2f} mm, Lap time = {len(traj)*ts:.2f} s")
    
    # Summary table
    print("\n[7/8] Performance Summary:")
    print(f"  {'Controller':<12} {'RMS [mm]':<12} {'Max [mm]':<12} {'Lap Time [s]':<15}")
    print(f"  {'-'*51}")
    for name in ['Stanley', 'LQR', 'MPC', 'DNN']:
        r = results[name]
        print(f"  {name:<12} {r['rms']*1000:<12.2f} {r['max']*1000:<12.2f} {len(r['trajectory'])*ts:<15.2f}")
    
    # Visualization
    print("\n[8/8] Creating visualizations...")
    plot_all_controllers(results, x_ref_array, y_ref_array, initial_state, ts)
    
    print("\n" + "="*70)
    print(" Comparison Complete!")
    print("="*70)
    print(f"\nVisualization saved to: outputs/all_controllers_comparison.png")


if __name__ == "__main__":
    main()
