"""
Simulate Stanley controller tracking the reference path for a complete lap.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.stanley_controller import StanleyController
from utils.reference import generate_ellipse_reference, find_projection_index


def simulate_stanley_complete_lap():
    """Simulate Stanley controller for a complete lap around the ellipse."""
    print("="*70)
    print(" Stanley Controller: Complete Lap Simulation")
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
    
    # Initialize Stanley controller
    dt = 0.03
    v_ref = 0.2
    stanley = StanleyController(v_ref=v_ref, dt=dt, k_gain=10.0, k_soft=0.01)
    
    # Initial state (start at reference with small offset)
    initial_state = np.array([x_ref[0] + 0.1, y_ref[0] + 0.05, theta_ref[0] + 0.1])
    state = initial_state.copy()
    
    # Simulation parameters
    max_steps = 5000
    lap_completion_threshold = 0.1
    min_steps_for_lap = 1000
    
    # Storage
    trajectory = [state.copy()]
    controls = []
    errors = []
    
    # Track starting position
    start_pos = state[:2]
    completed_lap = False
    
    print(f"\n[1/3] Starting simulation...")
    print(f"      Initial state: x={state[0]:.3f}, y={state[1]:.3f}, theta={state[2]:.3f}")
    print(f"      Reference start: x={x_ref[0]:.3f}, y={y_ref[0]:.3f}, theta={theta_ref[0]:.3f}")
    
    for step in range(max_steps):
        # Find closest reference point
        idx = find_projection_index(state, x_ref, y_ref)
        
        # Get reference state with curvature
        path_state = np.array([x_ref[idx], y_ref[idx], theta_ref[idx], curvature[idx]])
        
        # Compute control
        u = stanley.get_control(state, path_state)
        controls.append(u)
        
        # Calculate tracking error
        error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
        errors.append(error)
        
        # Apply kinematic unicycle dynamics
        v, omega = u[0], u[1]
        state = np.array([
            state[0] + dt * v * np.cos(state[2]),
            state[1] + dt * v * np.sin(state[2]),
            state[2] + dt * omega
        ])
        
        # Normalize angle
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        
        trajectory.append(state.copy())
        
        # Check for lap completion
        if step > min_steps_for_lap:
            dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
            if dist_to_start < lap_completion_threshold:
                completed_lap = True
                print(f"      Lap completed at step {step + 1}")
                break
    
    trajectory = np.array(trajectory)
    controls = np.array(controls)
    errors = np.array(errors)
    times = np.arange(len(controls)) * dt
    
    print(f"\n[2/3] Simulation complete!")
    print(f"      Steps: {len(trajectory)}")
    print(f"      Duration: {times[-1]:.2f} s")
    print(f"      Lap completed: {completed_lap}")
    print(f"      Mean tracking error: {np.mean(errors)*1000:.2f} mm")
    print(f"      Max tracking error: {np.max(errors)*1000:.2f} mm")
    
    # =========================================
    # Visualization
    # =========================================
    print(f"\n[3/3] Generating visualization...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Full trajectory
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Reference')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Stanley Trajectory', alpha=0.8)
    ax1.plot(initial_state[0], initial_state[1], 'go', markersize=12, label='Start', markeredgecolor='black')
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
    
    # Plot 3: Linear velocity
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times, controls[:, 0], 'g-', linewidth=2)
    ax3.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax3.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Linear Velocity [m/s]', fontsize=11, fontweight='bold')
    ax3.set_title('Linear Velocity', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 0.5])
    
    # Plot 4: Angular velocity
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(times, controls[:, 1], 'm-', linewidth=2)
    ax4.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Angular Velocity [rad/s]', fontsize=11, fontweight='bold')
    ax4.set_title('Angular Velocity', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Heading error
    ax5 = plt.subplot(2, 3, 5)
    heading_errors = []
    for i in range(len(controls)):
        state = trajectory[i]
        idx = find_projection_index(state, x_ref, y_ref)
        e_theta = state[2] - theta_ref[idx]
        e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))
        heading_errors.append(e_theta)
    ax5.plot(times, heading_errors, 'c-', linewidth=2)
    ax5.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Heading Error [rad]', fontsize=11, fontweight='bold')
    ax5.set_title('Heading Error', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(errors * 1000, bins=50, edgecolor='black', alpha=0.7)
    ax6.axvline(x=np.mean(errors)*1000, color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors)*1000:.2f} mm')
    ax6.set_xlabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Stanley Controller: Complete Lap Performance', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/stanley_complete_lap.png', dpi=200, bbox_inches='tight')
    
    print(f"      Visualization saved to 'outputs/stanley_complete_lap.png'")
    
    print("\n" + "="*70)
    print(" ✓ Stanley Simulation Complete!")
    print("="*70)
    
    return trajectory, controls, errors


if __name__ == "__main__":
    simulate_stanley_complete_lap()
