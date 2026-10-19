"""
Benchmark MPC timing with release build and warm starts.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.mpc_controller import MPCController
from utils.reference import generate_ellipse_reference, find_projection_index


def benchmark_mpc_timing():
    """Benchmark MPC solver timing with release build and warm starts."""
    print("="*70)
    print(" MPC Timing Benchmark (Release Build + Warm Starts)")
    print("="*70)
    
    # Generate reference trajectory
    total_points = 10000
    t_param, x_ref, y_ref, theta_ref = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    
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
    
    print(f"\n[1/2] Building MPC optimizer (release mode)...")
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    mpc.build_optimizer(build_dir="my_optimizers")
    print("      ✓ Release build complete")
    
    # Add optimizer to path
    sys.path.insert(1, "my_optimizers/path_following_aug")
    
    # Initial state
    initial_geometric_state = np.array([x_ref[0] + 0.1, y_ref[0] + 0.05, theta_ref[0] + 0.1])
    idx_closest = find_projection_index(initial_geometric_state, x_ref, y_ref)
    theta_p_initial = t_param[idx_closest]
    initial_aug_state = np.concatenate([initial_geometric_state, [theta_p_initial]])
    
    state = initial_aug_state.copy()
    
    print(f"\n[2/2] Benchmarking solver timing (100 steps with warm starts)...")
    
    solve_times = []
    u_prev = [0.01] * (nu * N)
    
    for step in range(100):
        # Get reference horizon
        theta_p_current = state[3]
        theta_ref_horizon_p = theta_p_current + v_ref * np.arange(N) * ts
        
        x_ref_horizon = np.interp(theta_ref_horizon_p, t_param, x_ref)
        y_ref_horizon = np.interp(theta_ref_horizon_p, t_param, y_ref)
        theta_ref_unwrapped = np.unwrap(theta_ref)
        theta_ref_horizon = np.interp(theta_ref_horizon_p, t_param, theta_ref_unwrapped)
        theta_ref_horizon = (theta_ref_horizon + np.pi) % (2 * np.pi) - np.pi
        
        # Solve MPC with warm start
        p_vector = np.concatenate([state, x_ref_horizon, y_ref_horizon, theta_ref_horizon])
        
        start_time = time.time()
        result = mpc.solver.run(p=p_vector, initial_guess=u_prev)
        solve_time = (time.time() - start_time) * 1000  # Convert to ms
        
        solve_times.append(solve_time)
        
        # Update warm start for next iteration
        if result.exit_status == 'Converged':
            u_prev = result.solution
        else:
            u_prev = [0.01] * (nu * N)
        
        # Apply dynamics
        u = u_prev[0:nu]
        s, omega, v_path = u[0], u[1], u[2]
        state = np.array([
            state[0] + ts * s * np.cos(state[2]),
            state[1] + ts * s * np.sin(state[2]),
            state[2] + ts * omega,
            state[3] + ts * v_path
        ])
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        state[3] = state[3] % (2 * np.pi)
    
    solve_times = np.array(solve_times)
    
    print(f"\nResults:")
    print(f"  Mean solve time: {np.mean(solve_times):.2f} ms")
    print(f"  Median solve time: {np.median(solve_times):.2f} ms")
    print(f"  Min solve time: {np.min(solve_times):.2f} ms")
    print(f"  Max solve time: {np.max(solve_times):.2f} ms")
    print(f"  Std dev: {np.std(solve_times):.2f} ms")
    
    print("\n" + "="*70)
    print(" Benchmark Complete!")
    print("="*70)


if __name__ == "__main__":
    benchmark_mpc_timing()
