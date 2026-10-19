"""
Visualize MPC control outputs along the full ellipse cycle.
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
from utils.reference import generate_ellipse_reference, find_projection_index


def visualize_mpc_controls():
    """Visualize MPC control outputs along the full ellipse cycle."""
    print("="*70)
    print(" MPC Control Outputs Visualization")
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
    
    # Initialize MPC controller
    nu = 3
    nx_aug = 4
    N = 70
    ts = 0.03
    q = 100
    qtheta = 10
    r_control = 0.1
    r_v = 0.001
    v_ref = 0.2
    
    print(f"\n[1/3] Building MPC optimizer...")
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    mpc.build_optimizer(build_dir="my_optimizers")
    print("      ✓ MPC ready")
    
    # Initial state
    initial_geometric_state = np.array([x_ref[0] + 0.1, y_ref[0] + 0.05, theta_ref[0] + 0.1])
    idx_closest = find_projection_index(initial_geometric_state, x_ref, y_ref)
    theta_p_initial = t_param[idx_closest]
    initial_aug_state = np.concatenate([initial_geometric_state, [theta_p_initial]])
    
    state = initial_aug_state.copy()
    
    max_steps = 3000
    min_steps_for_lap = 1000
    lap_completion_threshold = 0.1
    start_pos = state[:2]
    
    # Storage
    trajectory = [state.copy()]
    controls = []
    curvatures = []
    path_params = []
    errors = []
    
    print(f"\n[2/3] Simulating MPC for full lap...")
    
    for step in range(max_steps):
        # Get reference horizon
        theta_p_current = state[3]
        theta_ref_horizon_p = theta_p_current + v_ref * np.arange(N) * ts
        
        x_ref_horizon = np.interp(theta_ref_horizon_p, t_param, x_ref)
        y_ref_horizon = np.interp(theta_ref_horizon_p, t_param, y_ref)
        theta_ref_unwrapped = np.unwrap(theta_ref)
        theta_ref_horizon = np.interp(theta_ref_horizon_p, t_param, theta_ref_unwrapped)
        theta_ref_horizon = (theta_ref_horizon + np.pi) % (2 * np.pi) - np.pi
        
        # Get control from MPC
        u = mpc.get_first_control(state, x_ref_horizon, y_ref_horizon, theta_ref_horizon)
        controls.append(u)
        
        # Store additional info
        idx = find_projection_index(state, x_ref, y_ref)
        curvatures.append(curvature[idx])
        path_params.append(state[3])
        
        error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
        errors.append(error)
        
        # Apply dynamics
        s, omega, v_path = u[0], u[1], u[2]
        state = np.array([
            state[0] + ts * s * np.cos(state[2]),
            state[1] + ts * s * np.sin(state[2]),
            state[2] + ts * omega,
            state[3] + ts * v_path
        ])
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        state[3] = state[3] % (2 * np.pi)
        
        trajectory.append(state.copy())
        
        # Check lap completion
        if step > min_steps_for_lap:
            dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
            if dist_to_start < lap_completion_threshold:
                print(f"      Lap completed at step {step + 1}")
                break
    
    trajectory = np.array(trajectory)
    controls = np.array(controls)
    curvatures = np.array(curvatures)
    path_params = np.array(path_params)
    errors = np.array(errors)
    times = np.arange(len(controls)) * ts
    
    print(f"      Simulation complete: {len(controls)} steps")
    print(f"      Mean tracking error: {np.mean(errors)*1000:.2f} mm")
    
    # =========================================
    # Visualization
    # =========================================
    print(f"\n[3/3] Generating visualization...")
    
    fig = plt.figure(figsize=(20, 14))
    
    # Plot 1: Trajectory with color-coded control
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Reference')
    scatter = ax1.scatter(trajectory[:-1, 0], trajectory[:-1, 1], 
                        c=controls[:, 0], cmap='plasma', s=10, alpha=0.6, vmin=0.02, vmax=0.5)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'rs', markersize=12, label='End')
    ax1.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax1.set_title('Trajectory (colored by forward speed)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Forward Speed [m/s]', fontsize=10)
    
    # Plot 2: Forward speed vs time
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(times, controls[:, 0], 'b-', linewidth=2)
    ax2.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax2.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Forward Speed (s) [m/s]', fontsize=11, fontweight='bold')
    ax2.set_title('Forward Speed vs Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 4.5])
    
    # Plot 3: Angular velocity vs time
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(times, controls[:, 1], 'r-', linewidth=2)
    ax3.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Angular Velocity (ω) [rad/s]', fontsize=11, fontweight='bold')
    ax3.set_title('Angular Velocity vs Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Path parameter velocity vs time
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(times, controls[:, 2], 'g-', linewidth=2)
    ax4.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax4.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Path Velocity (v) [rad/s]', fontsize=11, fontweight='bold')
    ax4.set_title('Path Parameter Velocity vs Time', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 0.5])
    
    # Plot 5: Forward speed vs path parameter
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(path_params, controls[:, 0], 'b-', linewidth=2)
    ax5.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax5.set_xlabel('Path Parameter θ_p [rad]', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Forward Speed (s) [m/s]', fontsize=11, fontweight='bold')
    ax5.set_title('Forward Speed vs Path Parameter', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 2*np.pi])
    ax5.set_ylim([0, 4.5])
    
    # Plot 6: Angular velocity vs path parameter
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(path_params, controls[:, 1], 'r-', linewidth=2)
    ax6.set_xlabel('Path Parameter θ_p [rad]', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Angular Velocity (ω) [rad/s]', fontsize=11, fontweight='bold')
    ax6.set_title('Angular Velocity vs Path Parameter', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, 2*np.pi])
    
    # Plot 7: Path parameter velocity vs path parameter
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(path_params, controls[:, 2], 'g-', linewidth=2)
    ax7.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax7.set_xlabel('Path Parameter θ_p [rad]', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Path Velocity (v) [rad/s]', fontsize=11, fontweight='bold')
    ax7.set_title('Path Velocity vs Path Parameter', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, 2*np.pi])
    ax7.set_ylim([0, 0.5])
    
    # Plot 8: Controls vs curvature
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(curvatures, controls[:, 0], c=times, cmap='viridis', s=20, alpha=0.6, label='Forward Speed')
    ax8.scatter(curvatures, controls[:, 1], c=times, cmap='viridis', s=20, alpha=0.6, marker='s', label='Angular Vel')
    ax8.set_xlabel('Path Curvature [1/m]', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Control Value', fontsize=11, fontweight='bold')
    ax8.set_title('Controls vs Curvature', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Tracking error vs curvature
    ax9 = plt.subplot(3, 3, 9)
    ax9.scatter(curvatures, errors * 1000, c=times, cmap='viridis', s=20, alpha=0.6)
    ax9.set_xlabel('Path Curvature [1/m]', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax9.set_title('Tracking Error vs Curvature', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('MPC Control Outputs Along Full Ellipse Cycle', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/mpc_controls_visualization.png', dpi=200, bbox_inches='tight')
    
    print(f"      Visualization saved to 'outputs/mpc_controls_visualization.png'")
    
    # Print statistics
    print("\n" + "="*70)
    print(" Control Statistics")
    print("="*70)
    print(f"\nForward Speed (s):")
    print(f"  Mean: {np.mean(controls[:, 0]):.3f} m/s")
    print(f"  Std:  {np.std(controls[:, 0]):.3f} m/s")
    print(f"  Min:  {np.min(controls[:, 0]):.3f} m/s")
    print(f"  Max:  {np.max(controls[:, 0]):.3f} m/s")
    
    print(f"\nAngular Velocity (ω):")
    print(f"  Mean: {np.mean(controls[:, 1]):.3f} rad/s")
    print(f"  Std:  {np.std(controls[:, 1]):.3f} rad/s")
    print(f"  Min:  {np.min(controls[:, 1]):.3f} rad/s")
    print(f"  Max:  {np.max(controls[:, 1]):.3f} rad/s")
    
    print(f"\nPath Velocity (v):")
    print(f"  Mean: {np.mean(controls[:, 2]):.3f} rad/s")
    print(f"  Std:  {np.std(controls[:, 2]):.3f} rad/s")
    print(f"  Min:  {np.min(controls[:, 2]):.3f} rad/s")
    print(f"  Max:  {np.max(controls[:, 2]):.3f} rad/s")
    
    print("\n" + "="*70)
    print(" ✓ Visualization Complete!")
    print("="*70)


if __name__ == "__main__":
    visualize_mpc_controls()
