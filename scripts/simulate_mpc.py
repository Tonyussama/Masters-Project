"""
Simulate MPC controller tracking the reference path for a complete lap.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add system site-packages for pkg_resources
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.mpc_controller import MPCController
from utils.reference import generate_ellipse_reference, find_projection_index


def simulate_mpc_complete_lap():
    """Simulate MPC controller for a complete lap around the ellipse."""
    print("="*70)
    print(" MPC Controller: Complete Lap Simulation")
    print("="*70)
    
    # Generate reference trajectory
    total_points = 10000
    t_param, x_ref, y_ref, theta_ref = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    
    # Calculate path curvature
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    
    # Initialize MPC controller
    nu = 3  # [s, omega, v]
    nx_aug = 4  # [x, y, theta, theta_p]
    N = 70
    ts = 0.03
    q = 100
    qtheta = 10
    r_control = 0.1
    r_v = 0.001
    v_ref = 0.2
    
    print(f"\n[1/4] Building MPC optimizer...")
    print(f"      Horizon: {N} steps")
    print(f"      Time step: {ts} s")
    
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    
    try:
        solver = mpc.build_optimizer(build_dir="my_optimizers")
        print("      MPC optimizer ready!")
    except Exception as e:
        print(f"      Error building optimizer: {e}")
        print("      Using pre-built optimizer if available...")
        # Try to load pre-built optimizer
        import sys
        sys.path.insert(0, "my_optimizers/path_following_aug")
        try:
            import path_following_aug
            mpc.solver = path_following_aug.solver()
            mpc.built = True
            print("      Loaded pre-built optimizer!")
        except:
            print("      No pre-built optimizer found. Exiting.")
            return
    
    # Initial state (start at reference with small offset)
    initial_geometric_state = np.array([x_ref[0] + 0.1, y_ref[0] + 0.05, theta_ref[0] + 0.1])
    
    # Find initial path parameter
    idx_closest = find_projection_index(initial_geometric_state, x_ref, y_ref)
    theta_p_initial = t_param[idx_closest]
    initial_aug_state = np.concatenate([initial_geometric_state, [theta_p_initial]])
    
    state = initial_aug_state.copy()
    
    # Simulation parameters
    max_steps = 3000
    lap_completion_threshold = 0.1
    min_steps_for_lap = 1000
    
    # Storage
    trajectory = [state.copy()]
    controls = []
    errors = []
    
    # Track starting position
    start_pos = state[:2]
    completed_lap = False
    
    print(f"\n[2/4] Starting simulation...")
    print(f"      Initial state: x={state[0]:.3f}, y={state[1]:.3f}, theta={state[2]:.3f}, theta_p={state[3]:.3f}")
    print(f"      Reference start: x={x_ref[0]:.3f}, y={y_ref[0]:.3f}, theta={theta_ref[0]:.3f}")
    
    for step in range(max_steps):
        # Get reference horizon based on current theta_p
        theta_p_current = state[3]
        theta_ref_horizon_p = theta_p_current + v_ref * np.arange(N) * ts
        
        # Interpolate reference
        x_ref_horizon = np.interp(theta_ref_horizon_p, t_param, x_ref)
        y_ref_horizon = np.interp(theta_ref_horizon_p, t_param, y_ref)
        theta_ref_unwrapped = np.unwrap(theta_ref)
        theta_ref_horizon = np.interp(theta_ref_horizon_p, t_param, theta_ref_unwrapped)
        theta_ref_horizon = (theta_ref_horizon + np.pi) % (2 * np.pi) - np.pi
        
        # Get first control from MPC
        try:
            u = mpc.get_first_control(state, x_ref_horizon, y_ref_horizon, theta_ref_horizon)
        except Exception as e:
            print(f"      MPC solve failed at step {step}: {e}")
            # Use fallback control
            u = np.array([v_ref, 0.0, v_ref])
        
        controls.append(u)
        
        # Calculate tracking error
        idx = find_projection_index(state, x_ref, y_ref)
        error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
        errors.append(error)
        
        # Apply augmented dynamics
        s, omega, v_path = u[0], u[1], u[2]
        state = np.array([
            state[0] + ts * s * np.cos(state[2]),
            state[1] + ts * s * np.sin(state[2]),
            state[2] + ts * omega,
            state[3] + ts * v_path
        ])
        
        # Normalize angle and path parameter
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        state[3] = state[3] % (2 * np.pi)
        
        trajectory.append(state.copy())
        
        # Check for lap completion
        if step > min_steps_for_lap:
            dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
            if dist_to_start < lap_completion_threshold:
                completed_lap = True
                print(f"      Lap completed at step {step + 1}")
                break
        
        if (step + 1) % 100 == 0:
            print(f"      Step {step + 1}/{max_steps}, Error: {error*1000:.2f} mm")
    
    trajectory = np.array(trajectory)
    controls = np.array(controls)
    errors = np.array(errors)
    times = np.arange(len(controls)) * ts
    
    print(f"\n[3/4] Simulation complete!")
    print(f"      Steps: {len(trajectory)}")
    print(f"      Duration: {times[-1]:.2f} s")
    print(f"      Lap completed: {completed_lap}")
    print(f"      Mean tracking error: {np.mean(errors)*1000:.2f} mm")
    print(f"      Max tracking error: {np.max(errors)*1000:.2f} mm")
    
    # =========================================
    # Visualization
    # =========================================
    print(f"\n[4/4] Generating visualization...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Full trajectory
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Reference')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='MPC Trajectory', alpha=0.8)
    ax1.plot(initial_aug_state[0], initial_aug_state[1], 'go', markersize=12, label='Start', markeredgecolor='black')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'rs', markersize=12, label='End', markeredgecolor='black')
    ax1.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax1.set_title('Complete Lap Trajectory', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Tracking error over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(times, errors * 1000, 'r-', linewidth=2)
    ax2.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax2.set_title('Tracking Error Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Forward speed (s)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times, controls[:, 0], 'g-', linewidth=2)
    ax3.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax3.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Forward Speed (s) [m/s]', fontsize=11, fontweight='bold')
    ax3.set_title('Forward Speed', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 4.5])
    
    # Plot 4: Angular velocity
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(times, controls[:, 1], 'm-', linewidth=2)
    ax4.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Angular Velocity (ω) [rad/s]', fontsize=11, fontweight='bold')
    ax4.set_title('Angular Velocity', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Path parameter velocity
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(times, controls[:, 2], 'c-', linewidth=2)
    ax5.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax5.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Path Velocity (v) [rad/s]', fontsize=11, fontweight='bold')
    ax5.set_title('Path Parameter Velocity', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 0.5])
    
    # Plot 6: Error distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(errors * 1000, bins=50, edgecolor='black', alpha=0.7)
    ax6.axvline(x=np.mean(errors)*1000, color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors)*1000:.2f} mm')
    ax6.set_xlabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('MPC Controller: Complete Lap Performance', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/mpc_complete_lap.png', dpi=200, bbox_inches='tight')
    
    print(f"      Visualization saved to 'outputs/mpc_complete_lap.png'")
    
    print("\n" + "="*70)
    print(" ✓ MPC Simulation Complete!")
    print("="*70)
    
    return trajectory, controls, errors


if __name__ == "__main__":
    simulate_mpc_complete_lap()
