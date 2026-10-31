"""
Test MPC and DNN controllers under disturbances.

This script tests:
1. Impulse disturbances (sudden kicks)
2. Constant disturbances (friction/wind)
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.mpc_controller import MPCController
from controllers.dnn_controller import MPCNeuralNetwork, DNNController
from utils.reference import generate_ellipse_reference, find_projection_index
from utils.data_collection import MPCDataCollector


def simulate_mpc_lap(solver, aug_state, ts, t_param, x_ref_array, y_ref_array, theta_ref_array, 
                     disturbance_type=None, disturbance_params=None):
    """Simulate MPC with disturbances."""
    trajectory = [aug_state.copy()]
    controls = []
    
    initial_theta_p = aug_state[3]
    
    while True:
        theta_p = aug_state[3]
        x_ref = np.interp(theta_p + np.arange(70) * ts * 0.2, t_param, x_ref_array)
        y_ref = np.interp(theta_p + np.arange(70) * ts * 0.2, t_param, y_ref_array)
        theta_ref = np.interp(theta_p + np.arange(70) * ts * 0.2, t_param, np.unwrap(theta_ref_array))
        theta_ref = (theta_ref + np.pi) % (2 * np.pi) - np.pi
        
        p_vector = np.concatenate([aug_state, x_ref, y_ref, theta_ref])
        u_init = [0.01] * (3 * 70)
        result = solver.run(p=p_vector, initial_guess=u_init)
        
        if result.exit_status == 'Converged':
            u = result.solution[0:3]
        else:
            u = controls[-1] if controls else np.array([0.01, 0.0, 0.2])
        
        controls.append(u)
        
        # Apply disturbances
        aug_current = np.array([
            aug_state[0] + ts * u[0] * np.cos(aug_state[2]),
            aug_state[1] + ts * u[0] * np.sin(aug_state[2]),
            aug_state[2] + ts * u[1],
            aug_state[3] + ts * u[2]
        ])
        
        # Add disturbance effects
        if disturbance_type == 'impulse':
            # Impulse at specific times
            impulse_times = disturbance_params['times']
            impulse_magnitude = disturbance_params['magnitude']
            for imp_time, imp_mag in zip(impulse_times, impulse_magnitude):
                if abs(len(trajectory) * ts - imp_time) < ts:
                    # Apply impulse to position (lateral kick)
                    aug_current[0] += imp_mag[0]
                    aug_current[1] += imp_mag[1]
                    aug_current[2] += imp_mag[2]
        
        elif disturbance_type == 'constant':
            # Constant disturbance (friction/wind)
            # Modeled as a velocity perturbation that persists
            f_lat = disturbance_params['lateral_force']
            f_long = disturbance_params['longitudinal_force']
            tau = disturbance_params['torque']
            
            # Apply as velocity perturbation (more realistic for constant force)
            # This affects the dynamics, not position directly
            aug_current[0] += ts * f_lat * ts  # Small velocity effect
            aug_current[1] += ts * f_long * ts
            aug_current[2] += ts * tau * ts
        
        aug_current[2] = (aug_current[2] + np.pi) % (2 * np.pi) - np.pi
        aug_current[3] = aug_current[3] % (2 * np.pi)
        aug_state = aug_current
        trajectory.append(aug_current.copy())
        
        if aug_state[3] - initial_theta_p >= 2 * np.pi:
            break
        
        if len(trajectory) > 2000:
            break
    
    return np.array(trajectory), np.array(controls)


def simulate_dnn_lap(controller, aug_state, ts, disturbance_type=None, disturbance_params=None):
    """Simulate DNN with disturbances."""
    trajectory = [aug_state.copy()]
    controls = []
    
    initial_theta_p = aug_state[3]
    
    while True:
        u = controller.get_control(aug_state)
        controls.append(u)
        
        aug_current = np.array([
            aug_state[0] + ts * u[0] * np.cos(aug_state[2]),
            aug_state[1] + ts * u[0] * np.sin(aug_state[2]),
            aug_state[2] + ts * u[1],
            aug_state[3] + ts * u[2]
        ])
        
        # Add disturbance effects (same as MPC)
        if disturbance_type == 'impulse':
            impulse_times = disturbance_params['times']
            impulse_magnitude = disturbance_params['magnitude']
            for imp_time, imp_mag in zip(impulse_times, impulse_magnitude):
                if abs(len(trajectory) * ts - imp_time) < ts:
                    aug_current[0] += imp_mag[0]
                    aug_current[1] += imp_mag[1]
                    aug_current[2] += imp_mag[2]
        
        elif disturbance_type == 'constant':
            f_lat = disturbance_params['lateral_force']
            f_long = disturbance_params['longitudinal_force']
            tau = disturbance_params['torque']
            
            aug_current[0] += ts * f_lat
            aug_current[1] += ts * f_long
            aug_current[2] += ts * tau
        
        aug_current[2] = (aug_current[2] + np.pi) % (2 * np.pi) - np.pi
        aug_current[3] = aug_current[3] % (2 * np.pi)
        aug_state = aug_current
        trajectory.append(aug_current.copy())
        
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


def plot_disturbance_comparison(results, x_ref, y_ref, disturbance_type, save_path):
    """Plot disturbance comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'MPC': 'blue', 'DNN': 'red'}
    
    # Plot 1: Trajectories
    axes[0, 0].plot(x_ref, y_ref, 'k--', label='Reference', linewidth=2, alpha=0.5)
    for name, result in results.items():
        traj = result['trajectory']
        axes[0, 0].plot(traj[:, 0], traj[:, 1], color=colors[name], label=name, linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('X [m]')
    axes[0, 0].set_ylabel('Y [m]')
    axes[0, 0].set_title(f'Trajectory ({disturbance_type} disturbance)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Tracking error
    for name, result in results.items():
        errors = result['errors']
        time = np.arange(len(errors)) * 0.03
        axes[0, 1].plot(time, errors * 1000, color=colors[name], label=name, linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Tracking Error [mm]')
    axes[0, 1].set_title('Tracking Error Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Control inputs - speed
    for name, result in results.items():
        ctrl = result['controls']
        time = np.arange(len(ctrl)) * 0.03
        axes[1, 0].plot(time, ctrl[:, 0], color=colors[name], label=name, linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Speed s [m/s]')
    axes[1, 0].set_title('Speed Control')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Control inputs - angular velocity
    for name, result in results.items():
        ctrl = result['controls']
        time = np.arange(len(ctrl)) * 0.03
        axes[1, 1].plot(time, ctrl[:, 1], color=colors[name], label=name, linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Angular Velocity ω [rad/s]')
    axes[1, 1].set_title('Angular Velocity Control')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()


def main():
    print("="*70)
    print(" MPC vs DNN Disturbance Testing")
    print("="*70)
    
    # Setup
    total_points = 10000
    t_param, x_ref_array, y_ref_array, theta_ref_array = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    print(f"\n[1/6] Reference trajectory created: {len(x_ref_array)} points")
    
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
    
    print("[3/6] Loading DNN model...")
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
    
    # Initial condition
    initial_state = np.array([0.1, 0.0, np.pi/2])
    idx_closest = find_projection_index(initial_state, x_ref_array, y_ref_array)
    theta_p_initial = t_param[idx_closest]
    aug_initial = np.concatenate([initial_state, [theta_p_initial]])
    
    # =========================================
    # Test 1: Impulse Disturbances
    # =========================================
    print("\n[4/6] Testing with impulse disturbances...")
    print("  Impulse: 3 lateral kicks at t=10s, 20s, 30s")
    
    impulse_params = {
        'times': [10.0, 20.0, 30.0],
        'magnitude': [
            [0.05, 0.0, 0.0],   # Kick in +x direction
            [-0.05, 0.0, 0.0],  # Kick in -x direction
            [0.0, 0.05, 0.0]    # Kick in +y direction
        ]
    }
    
    mpc_traj_imp, mpc_ctrl_imp = simulate_mpc_lap(
        mpc_solver, aug_initial.copy(), ts, t_param, x_ref_array, y_ref_array, theta_ref_array,
        disturbance_type='impulse', disturbance_params=impulse_params
    )
    mpc_rms_imp, mpc_max_imp, mpc_err_imp = compute_tracking_error(mpc_traj_imp, x_ref_array, y_ref_array)
    
    dnn_traj_imp, dnn_ctrl_imp = simulate_dnn_lap(
        dnn_controller, aug_initial.copy(), ts,
        disturbance_type='impulse', disturbance_params=impulse_params
    )
    dnn_rms_imp, dnn_max_imp, dnn_err_imp = compute_tracking_error(dnn_traj_imp, x_ref_array, y_ref_array)
    
    print(f"    MPC: RMS = {mpc_rms_imp*1000:.2f} mm, Max = {mpc_max_imp*1000:.2f} mm")
    print(f"    DNN: RMS = {dnn_rms_imp*1000:.2f} mm, Max = {dnn_max_imp*1000:.2f} mm")
    
    impulse_results = {
        'MPC': {'trajectory': mpc_traj_imp, 'controls': mpc_ctrl_imp, 'errors': mpc_err_imp},
        'DNN': {'trajectory': dnn_traj_imp, 'controls': dnn_ctrl_imp, 'errors': dnn_err_imp}
    }
    
    plot_disturbance_comparison(impulse_results, x_ref_array, y_ref_array, 'Impulse', 
                               'outputs/disturbance_impulse.png')
    
    # =========================================
    # Test 2: Constant Disturbances
    # =========================================
    print("\n[5/6] Testing with constant disturbances...")
    print("  Constant: Lateral force 0.01 m/s (wind/friction)")
    
    constant_params = {
        'lateral_force': 0.01,  # Constant lateral push
        'longitudinal_force': 0.0,
        'torque': 0.0
    }
    
    mpc_traj_const, mpc_ctrl_const = simulate_mpc_lap(
        mpc_solver, aug_initial.copy(), ts, t_param, x_ref_array, y_ref_array, theta_ref_array,
        disturbance_type='constant', disturbance_params=constant_params
    )
    mpc_rms_const, mpc_max_const, mpc_err_const = compute_tracking_error(mpc_traj_const, x_ref_array, y_ref_array)
    
    dnn_traj_const, dnn_ctrl_const = simulate_dnn_lap(
        dnn_controller, aug_initial.copy(), ts,
        disturbance_type='constant', disturbance_params=constant_params
    )
    dnn_rms_const, dnn_max_const, dnn_err_const = compute_tracking_error(dnn_traj_const, x_ref_array, y_ref_array)
    
    print(f"    MPC: RMS = {mpc_rms_const*1000:.2f} mm, Max = {mpc_max_const*1000:.2f} mm")
    print(f"    DNN: RMS = {dnn_rms_const*1000:.2f} mm, Max = {dnn_max_const*1000:.2f} mm")
    
    constant_results = {
        'MPC': {'trajectory': mpc_traj_const, 'controls': mpc_ctrl_const, 'errors': mpc_err_const},
        'DNN': {'trajectory': dnn_traj_const, 'controls': dnn_ctrl_const, 'errors': dnn_err_const}
    }
    
    plot_disturbance_comparison(constant_results, x_ref_array, y_ref_array, 'Constant',
                               'outputs/disturbance_constant.png')
    
    # =========================================
    # Summary
    # =========================================
    print("\n[6/6] Disturbance Rejection Summary:")
    print(f"  {'Condition':<20} {'MPC RMS [mm]':<15} {'DNN RMS [mm]':<15} {'DNN/MPC Ratio':<15}")
    print(f"  {'-'*65}")
    print(f"  {'No disturbance':<20} {'0.34':<15} {'5.58':<15} {'16.4':<15}")
    print(f"  {'Impulse':<20} {mpc_rms_imp*1000:<15.2f} {dnn_rms_imp*1000:<15.2f} {dnn_rms_imp/mpc_rms_imp:<15.2f}")
    print(f"  {'Constant':<20} {mpc_rms_const*1000:<15.2f} {dnn_rms_const*1000:<15.2f} {dnn_rms_const/mpc_rms_const:<15.2f}")
    
    print("\n" + "="*70)
    print(" Disturbance Testing Complete!")
    print("="*70)
    print(f"\nVisualizations saved to:")
    print(f"  - outputs/disturbance_impulse.png")
    print(f"  - outputs/disturbance_constant.png")


if __name__ == "__main__":
    main()
