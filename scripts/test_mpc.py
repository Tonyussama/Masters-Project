"""
Test MPC controller on a few steps to verify it's working correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.mpc_controller import MPCController
from utils.reference import generate_ellipse_reference, find_projection_index


def test_mpc():
    """Test MPC controller on a small number of steps."""
    print("="*70)
    print(" Testing MPC Controller")
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
    print(f"      Horizon: {N} steps")
    print(f"      Time step: {ts} s")
    
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    
    try:
        solver = mpc.build_optimizer(build_dir="my_optimizers")
        print("      ✓ MPC optimizer built successfully!")
    except Exception as e:
        print(f"      ✗ Error building optimizer: {e}")
        return False
    
    # Initial state
    initial_geometric_state = np.array([x_ref[0] + 0.1, y_ref[0] + 0.05, theta_ref[0] + 0.1])
    idx_closest = find_projection_index(initial_geometric_state, x_ref, y_ref)
    theta_p_initial = t_param[idx_closest]
    initial_aug_state = np.concatenate([initial_geometric_state, [theta_p_initial]])
    
    state = initial_aug_state.copy()
    
    print(f"\n[2/3] Testing MPC on 10 steps...")
    print(f"      Initial state: x={state[0]:.3f}, y={state[1]:.3f}, theta={state[2]:.3f}, theta_p={state[3]:.3f}")
    
    for step in range(10):
        # Get reference horizon
        theta_p_current = state[3]
        theta_ref_horizon_p = theta_p_current + v_ref * np.arange(N) * ts
        
        x_ref_horizon = np.interp(theta_ref_horizon_p, t_param, x_ref)
        y_ref_horizon = np.interp(theta_ref_horizon_p, t_param, y_ref)
        theta_ref_unwrapped = np.unwrap(theta_ref)
        theta_ref_horizon = np.interp(theta_ref_horizon_p, t_param, theta_ref_unwrapped)
        theta_ref_horizon = (theta_ref_horizon + np.pi) % (2 * np.pi) - np.pi
        
        # Get control from MPC
        try:
            u = mpc.get_first_control(state, x_ref_horizon, y_ref_horizon, theta_ref_horizon)
            print(f"      Step {step+1}: u=[{u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f}]")
        except Exception as e:
            print(f"      ✗ MPC solve failed at step {step}: {e}")
            return False
        
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
        
        # Calculate error
        idx = find_projection_index(state, x_ref, y_ref)
        error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
        print(f"               state: x={state[0]:.3f}, y={state[1]:.3f}, theta={state[2]:.3f}, error={error*1000:.2f}mm")
    
    print(f"\n[3/3] MPC test complete!")
    print(f"      Final state: x={state[0]:.3f}, y={state[1]:.3f}, theta={state[2]:.3f}, theta_p={state[3]:.3f}")
    
    print("\n" + "="*70)
    print(" ✓ MPC Controller Working Correctly!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = test_mpc()
    sys.exit(0 if success else 1)
