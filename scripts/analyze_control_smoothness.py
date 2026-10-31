"""
Analyze control smoothness and identify issues in DNN outputs.

This script:
1. Computes control smoothness metrics (rate of change)
2. Identifies chattering behavior
3. Analyzes ω underestimation at tips
4. Quantifies systematic bias
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


def simulate_mpc_lap(solver, aug_state, ts, t_param, x_ref_array, y_ref_array, theta_ref_array):
    """Simulate MPC trajectory until lap completion."""
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
        
        if aug_state[3] - initial_theta_p >= 2 * np.pi:
            break
        
        if len(trajectory) > 2000:
            break
    
    return np.array(trajectory), np.array(controls)


def simulate_dnn_lap(controller, aug_state, ts):
    """Simulate DNN trajectory until lap completion."""
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
        aug_current[2] = (aug_current[2] + np.pi) % (2 * np.pi) - np.pi
        aug_current[3] = aug_current[3] % (2 * np.pi)
        aug_state = aug_current
        trajectory.append(aug_current.copy())
        
        if aug_state[3] - initial_theta_p >= 2 * np.pi:
            break
        
        if len(trajectory) > 2000:
            break
    
    return np.array(trajectory), np.array(controls)


def compute_control_smoothness(controls, ts):
    """Compute control smoothness metrics."""
    # Rate of change (jerk)
    du_dt = np.diff(controls, axis=0) / ts
    
    # RMS of rate of change
    s_jerk = np.sqrt(np.mean(du_dt[:, 0]**2))
    omega_jerk = np.sqrt(np.mean(du_dt[:, 1]**2))
    
    # Maximum rate of change
    s_max_jerk = np.max(np.abs(du_dt[:, 0]))
    omega_max_jerk = np.max(np.abs(du_dt[:, 1]))
    
    # Chattering indicator (high-frequency oscillations)
    # Count sign changes in derivative
    s_sign_changes = np.sum(np.diff(np.sign(du_dt[:, 0])) != 0)
    omega_sign_changes = np.sum(np.diff(np.sign(du_dt[:, 1])) != 0)
    
    return {
        's_jerk': s_jerk,
        'omega_jerk': omega_jerk,
        's_max_jerk': s_max_jerk,
        'omega_max_jerk': omega_max_jerk,
        's_sign_changes': s_sign_changes,
        'omega_sign_changes': omega_sign_changes
    }


def analyze_tips(traj, controls, t_param):
    """Analyze behavior at ellipse tips (high curvature regions)."""
    # Find tip regions (where curvature is highest)
    # Ellipse tips are at t ≈ π/2 and 3π/2
    tip_indices = []
    
    for i, aug_state in enumerate(traj[:-1]):
        theta_p = aug_state[3]
        # Check if near tip (within 0.2 rad of π/2 or 3π/2)
        if (abs(theta_p - np.pi/2) < 0.2) or (abs(theta_p - 3*np.pi/2) < 0.2):
            tip_indices.append(i)
    
    if len(tip_indices) == 0:
        return None
    
    tip_omega = controls[tip_indices, 1]
    tip_speed = controls[tip_indices, 0]
    
    return {
        'omega_mean': np.mean(tip_omega),
        'omega_max': np.max(tip_omega),
        'omega_min': np.min(tip_omega),
        'speed_mean': np.mean(tip_speed),
        'count': len(tip_indices)
    }


def compute_systematic_bias(traj, x_ref, y_ref, t_param_local, theta_ref_array):
    """Compute systematic lateral bias."""
    # Compute signed lateral error (positive = left of path)
    lateral_errors = []
    
    for aug_state in traj[:-1]:
        idx = find_projection_index(aug_state[:2], x_ref, y_ref)
        x_path = x_ref[idx]
        y_path = y_ref[idx]
        theta_path = np.interp(t_param_local[idx], t_param_local, theta_ref_array)
        
        # Transform to path frame
        dx = aug_state[0] - x_path
        dy = aug_state[1] - y_path
        cos_theta = np.cos(-theta_path)
        sin_theta = np.sin(-theta_path)
        lateral_error = dx * sin_theta + dy * cos_theta
        lateral_errors.append(lateral_error)
    
    lateral_errors = np.array(lateral_errors)
    
    return {
        'mean_bias': np.mean(lateral_errors),
        'std_bias': np.std(lateral_errors),
        'max_bias': np.max(lateral_errors),
        'min_bias': np.min(lateral_errors)
    }


def main():
    print("="*70)
    print(" Control Smoothness and Bias Analysis")
    print("="*70)
    
    # Setup
    total_points = 10000
    t_param, x_ref_array, y_ref_array, theta_ref_array = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    
    nu = 3
    nx_aug = 4
    N = 70
    ts = 0.03
    q = 100
    qtheta = 10
    r_control = 0.1
    r_v = 0.001
    v_ref = 0.2
    
    print("\n[1/5] Building MPC controller...")
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    mpc_solver = mpc.build_optimizer(build_dir="my_optimizers")
    
    print("[2/5] Loading DNN model...")
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
    
    print("[3/5] Simulating controllers...")
    initial_state = np.array([0.1, 0.0, np.pi/2])
    idx_closest = find_projection_index(initial_state, x_ref_array, y_ref_array)
    theta_p_initial = t_param[idx_closest]
    aug_initial = np.concatenate([initial_state, [theta_p_initial]])
    
    mpc_traj, mpc_ctrl = simulate_mpc_lap(mpc_solver, aug_initial.copy(), ts, 
                                          t_param, x_ref_array, y_ref_array, theta_ref_array)
    dnn_traj, dnn_ctrl = simulate_dnn_lap(dnn_controller, aug_initial.copy(), ts)
    
    print("[4/5] Computing smoothness metrics...")
    mpc_smoothness = compute_control_smoothness(mpc_ctrl, ts)
    dnn_smoothness = compute_control_smoothness(dnn_ctrl, ts)
    
    print("\n  Control Smoothness Comparison:")
    print(f"  {'Metric':<25} {'MPC':<15} {'DNN':<15} {'Ratio (DNN/MPC)':<15}")
    print(f"  {'-'*70}")
    print(f"  {'Speed jerk (RMS)':<25} {mpc_smoothness['s_jerk']:<15.4f} {dnn_smoothness['s_jerk']:<15.4f} {dnn_smoothness['s_jerk']/mpc_smoothness['s_jerk']:<15.2f}")
    print(f"  {'Speed jerk (max)':<25} {mpc_smoothness['s_max_jerk']:<15.4f} {dnn_smoothness['s_max_jerk']:<15.4f} {dnn_smoothness['s_max_jerk']/mpc_smoothness['s_max_jerk']:<15.2f}")
    print(f"  {'Speed sign changes':<25} {mpc_smoothness['s_sign_changes']:<15} {dnn_smoothness['s_sign_changes']:<15} {dnn_smoothness['s_sign_changes']/mpc_smoothness['s_sign_changes']:<15.2f}")
    print(f"  {'ω jerk (RMS)':<25} {mpc_smoothness['omega_jerk']:<15.4f} {dnn_smoothness['omega_jerk']:<15.4f} {dnn_smoothness['omega_jerk']/mpc_smoothness['omega_jerk']:<15.2f}")
    print(f"  {'ω jerk (max)':<25} {mpc_smoothness['omega_max_jerk']:<15.4f} {dnn_smoothness['omega_max_jerk']:<15.4f} {dnn_smoothness['omega_max_jerk']/mpc_smoothness['omega_max_jerk']:<15.2f}")
    
    print("\n[5/5] Analyzing tips and bias...")
    mpc_tips = analyze_tips(mpc_traj, mpc_ctrl, t_param)
    dnn_tips = analyze_tips(dnn_traj, dnn_ctrl, t_param)
    
    if mpc_tips and dnn_tips:
        print("\n  Tip Behavior (High Curvature Regions):")
        print(f"  {'Metric':<20} {'MPC':<15} {'DNN':<15} {'Difference':<15}")
        print(f"  {'-'*65}")
        print(f"  {'ω mean [rad/s]':<20} {mpc_tips['omega_mean']:<15.4f} {dnn_tips['omega_mean']:<15.4f} {dnn_tips['omega_mean']-mpc_tips['omega_mean']:<15.4f}")
        print(f"  {'ω max [rad/s]':<20} {mpc_tips['omega_max']:<15.4f} {dnn_tips['omega_max']:<15.4f} {dnn_tips['omega_max']-mpc_tips['omega_max']:<15.4f}")
        print(f"  {'Speed mean [m/s]':<20} {mpc_tips['speed_mean']:<15.4f} {dnn_tips['speed_mean']:<15.4f} {dnn_tips['speed_mean']-mpc_tips['speed_mean']:<15.4f}")
    
    mpc_bias = compute_systematic_bias(mpc_traj, x_ref_array, y_ref_array, t_param, theta_ref_array)
    dnn_bias = compute_systematic_bias(dnn_traj, x_ref_array, y_ref_array, t_param, theta_ref_array)
    
    print("\n  Systematic Lateral Bias:")
    print(f"  {'Metric':<20} {'MPC':<15} {'DNN':<15} {'Difference':<15}")
    print(f"  {'-'*65}")
    print(f"  {'Mean bias [mm]':<20} {mpc_bias['mean_bias']*1000:<15.4f} {dnn_bias['mean_bias']*1000:<15.4f} {(dnn_bias['mean_bias']-mpc_bias['mean_bias'])*1000:<15.4f}")
    print(f"  {'Std bias [mm]':<20} {mpc_bias['std_bias']*1000:<15.4f} {dnn_bias['std_bias']*1000:<15.4f} {(dnn_bias['std_bias']-mpc_bias['std_bias'])*1000:<15.4f}")
    
    print("\n" + "="*70)
    print(" Analysis Complete!")
    print("="*70)
    print("\nKey Findings:")
    print(f"  1. Speed chattering: {dnn_smoothness['s_sign_changes']/mpc_smoothness['s_sign_changes']:.1f}x more sign changes")
    print(f"  2. ω underestimation at tips: {dnn_tips['omega_mean']-mpc_tips['omega_mean']:.4f} rad/s")
    print(f"  3. Systematic bias: {dnn_bias['mean_bias']*1000:.2f} mm")
    print("\nRecommendations:")
    print("  - Add smoothness penalty to loss function")
    print("  - Increase sampling density near tips")
    print("  - Add bias correction layer")


if __name__ == "__main__":
    main()
