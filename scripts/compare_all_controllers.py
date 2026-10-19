"""
Compare MPC, LQR, and Stanley controllers on the reference path.
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
from controllers.lqr_controller import LQRController
from utils.reference import generate_ellipse_reference, find_projection_index


def simulate_mpc_short(mpc, x_ref, y_ref, theta_ref, curvature, t_param, 
                      initial_state, dt=0.03, max_steps=500):
    """Simulate MPC for a shorter duration (for comparison)."""
    state = initial_state.copy()
    trajectory = [state.copy()]
    controls = []
    errors = []
    
    start_pos = state[:2]
    completed_lap = False
    min_steps_for_lap = 100
    lap_completion_threshold = 0.1
    
    for step in range(max_steps):
        # Get reference horizon
        theta_p_current = state[3]
        N = mpc.N
        v_ref = mpc.v_ref
        theta_ref_horizon_p = theta_p_current + v_ref * np.arange(N) * dt
        
        x_ref_horizon = np.interp(theta_ref_horizon_p, t_param, x_ref)
        y_ref_horizon = np.interp(theta_ref_horizon_p, t_param, y_ref)
        theta_ref_unwrapped = np.unwrap(theta_ref)
        theta_ref_horizon = np.interp(theta_ref_horizon_p, t_param, theta_ref_unwrapped)
        theta_ref_horizon = (theta_ref_horizon + np.pi) % (2 * np.pi) - np.pi
        
        # Get control from MPC
        try:
            u = mpc.get_first_control(state, x_ref_horizon, y_ref_horizon, theta_ref_horizon)
        except Exception as e:
            print(f"      MPC failed at step {step}: {e}")
            break
        
        controls.append(u)
        
        # Calculate error
        idx = find_projection_index(state, x_ref, y_ref)
        error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
        errors.append(error)
        
        # Apply dynamics
        s, omega, v_path = u[0], u[1], u[2]
        state = np.array([
            state[0] + dt * s * np.cos(state[2]),
            state[1] + dt * s * np.sin(state[2]),
            state[2] + dt * omega,
            state[3] + dt * v_path
        ])
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        state[3] = state[3] % (2 * np.pi)
        
        trajectory.append(state.copy())
        
        # Check lap completion
        if step > min_steps_for_lap:
            dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
            if dist_to_start < lap_completion_threshold:
                completed_lap = True
                break
    
    return np.array(trajectory), np.array(controls), np.array(errors), completed_lap


def simulate_lqr_short(lqr, x_ref, y_ref, theta_ref, curvature, 
                      initial_state, dt=0.03, max_steps=500):
    """Simulate LQR for comparison."""
    state = initial_state.copy()
    trajectory = [state.copy()]
    controls = []
    errors = []
    
    start_pos = state[:2]
    completed_lap = False
    min_steps_for_lap = 100
    lap_completion_threshold = 0.1
    
    for step in range(max_steps):
        idx = find_projection_index(state, x_ref, y_ref)
        path_state = np.array([x_ref[idx], y_ref[idx], theta_ref[idx], curvature[idx]])
        
        u = lqr.get_control(state, path_state)
        controls.append(u)
        
        error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
        errors.append(error)
        
        v, omega = u[0], u[1]
        state = np.array([
            state[0] + dt * v * np.cos(state[2]),
            state[1] + dt * v * np.sin(state[2]),
            state[2] + dt * omega
        ])
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        
        trajectory.append(state.copy())
        
        if step > min_steps_for_lap:
            dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
            if dist_to_start < lap_completion_threshold:
                completed_lap = True
                break
    
    return np.array(trajectory), np.array(controls), np.array(errors), completed_lap


def compare_all_controllers():
    """Compare MPC and LQR on the reference path."""
    print("="*70)
    print(" Controller Comparison: MPC vs LQR")
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
    max_steps = 3000  # Full lap simulation
    
    # Initial state
    initial_geometric_state = np.array([x_ref[0] + 0.1, y_ref[0] + 0.05, theta_ref[0] + 0.1])
    idx_closest = find_projection_index(initial_geometric_state, x_ref, y_ref)
    theta_p_initial = t_param[idx_closest]
    initial_aug_state = np.concatenate([initial_geometric_state, [theta_p_initial]])
    
    # Initialize controllers
    print(f"\n[1/4] Initializing controllers...")
    
    # MPC
    print("      Building MPC...")
    mpc = MPCController(nu=3, nx_aug=4, N=70, ts=dt, q=100, qtheta=10, 
                       r_control=0.1, r_v=0.001, v_ref=v_ref)
    mpc.build_optimizer(build_dir="my_optimizers")
    print("      ✓ MPC ready")
    
    # LQR
    lqr = LQRController(v_ref=v_ref, dt=dt)
    print("      ✓ LQR ready")
    
    # Simulate each controller
    print(f"\n[2/4] Simulating controllers ({max_steps} steps each)...")
    
    print("      Simulating MPC...")
    traj_mpc, ctrl_mpc, err_mpc, lap_mpc = simulate_mpc_short(
        mpc, x_ref, y_ref, theta_ref, curvature, t_param, 
        initial_aug_state, dt, max_steps
    )
    print(f"      MPC: Mean error={np.mean(err_mpc)*1000:.2f}mm, Lap={lap_mpc}")
    
    print("      Simulating LQR...")
    traj_lqr, ctrl_lqr, err_lqr, lap_lqr = simulate_lqr_short(
        lqr, x_ref, y_ref, theta_ref, curvature, 
        initial_geometric_state, dt, max_steps
    )
    print(f"      LQR: Mean error={np.mean(err_lqr)*1000:.2f}mm, Lap={lap_lqr}")
    
    # Visualization
    print(f"\n[3/4] Generating visualization...")
    
    fig = plt.figure(figsize=(20, 10))
    
    # Plot 1: All trajectories
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Reference')
    ax1.plot(traj_mpc[:, 0], traj_mpc[:, 1], 'b-', linewidth=2, label='MPC', alpha=0.8)
    ax1.plot(traj_lqr[:, 0], traj_lqr[:, 1], 'g-', linewidth=2, label='LQR', alpha=0.8)
    ax1.plot(initial_aug_state[0], initial_aug_state[1], 'ko', markersize=10, label='Start')
    ax1.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax1.set_title('Trajectories Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Tracking errors
    ax2 = plt.subplot(2, 3, 2)
    times_mpc = np.arange(len(err_mpc)) * dt
    times_lqr = np.arange(len(err_lqr)) * dt
    ax2.plot(times_mpc, err_mpc * 1000, 'b-', linewidth=2, label='MPC')
    ax2.plot(times_lqr, err_lqr * 1000, 'g-', linewidth=2, label='LQR')
    ax2.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax2.set_title('Tracking Error Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Linear velocity
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times_mpc, ctrl_mpc[:, 0], 'b-', linewidth=2, label='MPC')
    ax3.plot(times_lqr, ctrl_lqr[:, 0], 'g-', linewidth=2, label='LQR')
    ax3.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax3.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Linear Velocity [m/s]', fontsize=11, fontweight='bold')
    ax3.set_title('Linear Velocity Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Angular velocity
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(times_mpc, ctrl_mpc[:, 1], 'b-', linewidth=2, label='MPC')
    ax4.plot(times_lqr, ctrl_lqr[:, 1], 'g-', linewidth=2, label='LQR')
    ax4.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Angular Velocity [rad/s]', fontsize=11, fontweight='bold')
    ax4.set_title('Angular Velocity Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Path parameter velocity (MPC only)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(times_mpc, ctrl_mpc[:, 2], 'b-', linewidth=2, label='MPC')
    ax5.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax5.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Path Velocity [rad/s]', fontsize=11, fontweight='bold')
    ax5.set_title('Path Velocity (MPC only)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(err_mpc * 1000, bins=30, alpha=0.5, label='MPC', color='blue')
    ax6.hist(err_lqr * 1000, bins=30, alpha=0.5, label='LQR', color='green')
    ax6.set_xlabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Controller Comparison: MPC vs LQR', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/controller_comparison_all.png', dpi=200, bbox_inches='tight')
    
    print(f"      Visualization saved to 'outputs/controller_comparison_all.png'")
    
    # Summary table
    print(f"\n[4/4] Summary:")
    print("="*70)
    print(f"{'Controller':<12} {'Mean Error':<15} {'Max Error':<15} {'Lap':<10}")
    print("-" * 70)
    print(f"{'MPC':<12} {np.mean(err_mpc)*1000:<15.2f} {np.max(err_mpc)*1000:<15.2f} {str(lap_mpc):<10}")
    print(f"{'LQR':<12} {np.mean(err_lqr)*1000:<15.2f} {np.max(err_lqr)*1000:<15.2f} {str(lap_lqr):<10}")
    print("="*70)
    
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    compare_all_controllers()
