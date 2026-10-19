"""
Compare LQR and Stanley controllers on a more reasonable path.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.lqr_controller import LQRController
from controllers.stanley_controller import StanleyController
from utils.reference import generate_ellipse_reference, find_projection_index


def simulate_controller(controller, controller_name, x_ref, y_ref, theta_ref, curvature, 
                       initial_state, dt=0.03, max_steps=3000):
    """Simulate a controller for a complete lap."""
    state = initial_state.copy()
    trajectory = [state.copy()]
    controls = []
    errors = []
    
    start_pos = state[:2]
    lap_completion_threshold = 0.1
    min_steps_for_lap = 500
    completed_lap = False
    
    for step in range(max_steps):
        idx = find_projection_index(state, x_ref, y_ref)
        path_state = np.array([x_ref[idx], y_ref[idx], theta_ref[idx], curvature[idx]])
        
        u = controller.get_control(state, path_state)
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
    
    trajectory = np.array(trajectory)
    controls = np.array(controls)
    errors = np.array(errors)
    
    return trajectory, controls, errors, completed_lap


def compare_controllers():
    """Compare LQR and Stanley on different path widths."""
    print("="*70)
    print(" Controller Comparison: Different Path Widths")
    print("="*70)
    
    dt = 0.03
    v_ref = 0.2
    
    # Test different path widths
    widths = [0.2, 0.5, 1.0, 2.0]
    height = 4.0
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    results = []
    
    for i, width in enumerate(widths):
        print(f"\n[Width {width}m]")
        
        # Generate reference
        t_param, x_ref, y_ref, theta_ref = generate_ellipse_reference(
            total_points=10000, height=height, width=width
        )
        
        # Calculate curvature
        dx = np.gradient(x_ref)
        dy = np.gradient(y_ref)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        
        max_curv = np.max(np.abs(curvature))
        print(f"  Max curvature: {max_curv:.2f} 1/m")
        
        # Initial state
        initial_state = np.array([x_ref[0] + 0.1, y_ref[0] + 0.05, theta_ref[0] + 0.1])
        
        # Test LQR
        lqr = LQRController(v_ref=v_ref, dt=dt)
        traj_lqr, ctrl_lqr, err_lqr, lap_lqr = simulate_controller(
            lqr, "LQR", x_ref, y_ref, theta_ref, curvature, initial_state, dt
        )
        
        # Test Stanley
        stanley = StanleyController(v_ref=v_ref, dt=dt, k_gain=5.0, k_soft=0.1)
        traj_stanley, ctrl_stanley, err_stanley, lap_stanley = simulate_controller(
            stanley, "Stanley", x_ref, y_ref, theta_ref, curvature, initial_state, dt
        )
        
        results.append({
            'width': width,
            'max_curv': max_curv,
            'lqr': {'traj': traj_lqr, 'err': err_lqr, 'lap': lap_lqr},
            'stanley': {'traj': traj_stanley, 'err': err_stanley, 'lap': lap_stanley}
        })
        
        print(f"  LQR: Mean error={np.mean(err_lqr)*1000:.2f}mm, Lap={lap_lqr}")
        print(f"  Stanley: Mean error={np.mean(err_stanley)*1000:.2f}mm, Lap={lap_stanley}")
        
        # Plot trajectories
        ax = axes[0, i]
        ax.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Reference')
        ax.plot(traj_lqr[:, 0], traj_lqr[:, 1], 'b-', linewidth=2, label='LQR', alpha=0.7)
        ax.plot(traj_stanley[:, 0], traj_stanley[:, 1], 'r-', linewidth=2, label='Stanley', alpha=0.7)
        ax.set_xlabel('x [m]', fontsize=10)
        ax.set_ylabel('y [m]', fontsize=10)
        ax.set_title(f'Width={width}m (max κ={max_curv:.1f})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot errors
        ax_err = axes[1, i]
        times_lqr = np.arange(len(err_lqr)) * dt
        times_stanley = np.arange(len(err_stanley)) * dt
        ax_err.plot(times_lqr, err_lqr * 1000, 'b-', linewidth=2, label='LQR')
        ax_err.plot(times_stanley, err_stanley * 1000, 'r-', linewidth=2, label='Stanley')
        ax_err.set_xlabel('Time [s]', fontsize=10)
        ax_err.set_ylabel('Error [mm]', fontsize=10)
        ax_err.set_title('Tracking Error', fontsize=11, fontweight='bold')
        ax_err.legend(fontsize=8)
        ax_err.grid(True, alpha=0.3)
    
    plt.suptitle('Controller Comparison: Effect of Path Curvature', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/controller_comparison.png', dpi=200, bbox_inches='tight')
    
    print(f"\nVisualization saved to 'outputs/controller_comparison.png'")
    
    # Summary table
    print("\n" + "="*70)
    print(" Summary Table")
    print("="*70)
    print(f"{'Width':<10} {'Max Curv':<12} {'LQR Error':<15} {'Stanley Error':<15} {'LQR Lap':<10} {'Stanley Lap':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['width']:<10.1f} {r['max_curv']:<12.2f} {np.mean(r['lqr']['err'])*1000:<15.2f} "
              f"{np.mean(r['stanley']['err'])*1000:<15.2f} {str(r['lqr']['lap']):<10} {str(r['stanley']['lap']):<12}")
    
    print("\n" + "="*70)
    print(" ✓ Comparison Complete!")
    print("="*70)


if __name__ == "__main__":
    compare_controllers()
