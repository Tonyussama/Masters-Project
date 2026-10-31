"""
Data collection script for MPC dataset (CPU-only).
Run this on your laptop to collect data, then upload to Colab for training.

This script:
1. Sets up the MPC controller
2. Collects training data from MPC
3. Saves dataset to .npz file for later training
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.mpc_controller import MPCController
from utils.reference import generate_ellipse_reference
from utils.data_collection import MPCDataCollector


def main():
    print("="*70)
    print(" MPC Data Collection (CPU-only)")
    print("="*70)
    
    # =========================================
    # Setup Reference Trajectory
    # =========================================
    total_points = 10000
    t_param, x_ref_array, y_ref_array, theta_ref_array = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    print(f"\n[1/3] Reference trajectory created: {len(x_ref_array)} points")
    
    # =========================================
    # Build MPC Controller
    # =========================================
    nu = 3
    nx_aug = 4
    N = 70
    ts = 0.03
    q = 100
    qtheta = 10
    r_control = 0.1
    r_v = 0.001
    v_ref = 0.2
    
    print("[2/3] Building MPC optimizer...")
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    solver = mpc.build_optimizer(build_dir="my_optimizers")
    print("      MPC optimizer ready!")
    
    # =========================================
    # Collect Training Data
    # =========================================
    collector = MPCDataCollector(solver, t_param, x_ref_array, y_ref_array, 
                                 theta_ref_array, N, nu, ts, v_ref)
    
    # Use fewer anchors for laptop (2000 instead of 5000)
    # Still gives ~17k samples with 92.8% convergence
    states, references, controls, failure_info, block_ids = collector.collect_data_corridor(
        num_anchors=2000, offset_distance=0.1, 
        heading_error_range=np.pi/6, keep_failures=True
    )
    
    # Save dataset
    np.savez('data/mpc_dataset.npz',
             states=states,
             references=references,
             controls=controls,
             block_ids=block_ids,
             x_ref_array=x_ref_array,
             y_ref_array=y_ref_array,
             t_param=t_param,
             theta_ref_array=theta_ref_array,
             N=N, nu=nu, ts=ts)
    print("\n[3/3] Dataset saved to 'data/mpc_dataset.npz'")
    print(f"      Total samples: {len(states)}")
    print(f"      Failure rate: {failure_info['rate']*100:.1f}%")
    
    print("\n" + "="*70)
    print(" Data Collection Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Upload 'data/mpc_dataset.npz' to Google Colab")
    print("  2. Run the Colab training notebook")
    print("  3. Download the trained .pth model file")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
