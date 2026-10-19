"""
Test the corridor experiment data collection methodology.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.mpc_controller import MPCController
from utils.reference import generate_ellipse_reference, find_projection_index
from utils.data_collection import MPCDataCollector


def test_corridor_method():
    """Test the corridor experiment data collection."""
    print("="*70)
    print(" Testing Corridor Experiment Methodology")
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
    
    print(f"\n[1/3] Building MPC optimizer...")
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    solver = mpc.build_optimizer(build_dir="my_optimizers")
    print("      ✓ MPC ready")
    
    # Test corridor method with small number of anchors
    print(f"\n[2/3] Testing corridor method (10 anchors)...")
    collector = MPCDataCollector(solver, t_param, x_ref, y_ref, theta_ref, N, nu, ts, v_ref)
    states, references, controls = collector.collect_data_corridor(num_anchors=10, offset_distance=0.1)
    
    print(f"\n[3/3] Results:")
    print(f"      Collected {len(states)} samples")
    print(f"      Expected: 10 anchors × 9 seeds = 90 samples")
    print(f"      Success rate: {len(states)/90*100:.1f}%")
    
    # Verify data structure
    print(f"\n      State shape: {states.shape}")
    print(f"      Reference features shape: {references.shape}")
    print(f"      Controls shape: {controls.shape}")
    
    # Show some sample data
    print(f"\n      Sample states (first 3):")
    for i in range(min(3, len(states))):
        print(f"        Sample {i+1}: x={states[i,0]:.3f}, y={states[i,1]:.3f}, theta={states[i,2]:.3f}")
    
    print(f"\n      Sample controls (first 3):")
    for i in range(min(3, len(controls))):
        print(f"        Sample {i+1}: s={controls[i,0]:.3f}, omega={controls[i,1]:.3f}, v={controls[i,2]:.3f}")
    
    print("\n" + "="*70)
    print(" ✓ Corridor Method Test Complete!")
    print("="*70)


if __name__ == "__main__":
    test_corridor_method()
