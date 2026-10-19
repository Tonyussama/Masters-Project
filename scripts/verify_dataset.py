"""
Verify the collected dataset by printing samples and visualizing on ellipse.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.reference import generate_ellipse_reference, find_projection_index


def verify_dataset():
    """Verify the dataset by printing samples and visualizing."""
    print("="*70)
    print(" Dataset Verification")
    print("="*70)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    try:
        data = np.load('mpc_dataset.npz')
        states = data['states']
        references = data['references']
        controls = data['controls']
        print(f"      States shape: {states.shape}")
        print(f"      References shape: {references.shape}")
        print(f"      Controls shape: {controls.shape}")
        print(f"      Total samples: {len(states)}")
    except FileNotFoundError:
        print("      ERROR: Dataset not found. Run data collection first.")
        return
    
    # Print sample points
    print("\n[2/3] Sample points from dataset:")
    print("-" * 70)
    
    num_samples = 5
    for i in range(num_samples):
        print(f"\nSample {i+1}:")
        print(f"  State [x, y, theta]:")
        print(f"    x = {states[i, 0]:.4f} m")
        print(f"    y = {states[i, 1]:.4f} m")
        print(f"    theta = {states[i, 2]:.4f} rad")
        print(f"  Control [s, omega, v_path]:")
        print(f"    s = {controls[i, 0]:.4f} m/s")
        print(f"    omega = {controls[i, 1]:.4f} rad/s")
        print(f"    v_path = {controls[i, 2]:.4f} rad/s")
        print(f"  Reference features (first 10 of 42):")
        print(f"    {references[i, :10]}")
    
    # Statistics
    print("\n" + "-" * 70)
    print("Dataset Statistics:")
    print(f"  States mean: {states.mean(axis=0)}")
    print(f"  States std: {states.std(axis=0)}")
    print(f"  Controls mean: {controls.mean(axis=0)}")
    print(f"  Controls std: {controls.std(axis=0)}")
    
    # Control ranges
    print(f"\n  Control ranges:")
    print(f"    s: [{controls[:, 0].min():.4f}, {controls[:, 0].max():.4f}] m/s")
    print(f"    omega: [{controls[:, 1].min():.4f}, {controls[:, 1].max():.4f}] rad/s")
    print(f"    v_path: [{controls[:, 2].min():.4f}, {controls[:, 2].max():.4f}] rad/s")
    
    # Visualize on ellipse
    print("\n[3/3] Visualizing dataset on ellipse...")
    
    # Generate reference trajectory
    total_points = 10000
    t_param, x_ref, y_ref, theta_ref = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All states on ellipse
    ax1 = axes[0, 0]
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Reference')
    scatter = ax1.scatter(states[:, 0], states[:, 1], c=states[:, 2], 
                         cmap='hsv', s=5, alpha=0.6, label='States')
    ax1.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax1.set_title('All States in Dataset', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    plt.colorbar(scatter, ax=ax1, label='Heading [rad]')
    
    # Plot 2: Subsample of states (for clarity)
    ax2 = axes[0, 1]
    subsample_idx = np.random.choice(len(states), size=min(1000, len(states)), replace=False)
    ax2.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Reference')
    ax2.scatter(states[subsample_idx, 0], states[subsample_idx, 1], 
               c=states[subsample_idx, 2], cmap='hsv', s=10, alpha=0.7, label='States (subsample)')
    ax2.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax2.set_title('States (1000 Subsample)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Control distributions
    ax3 = axes[1, 0]
    ax3.hist(controls[:, 0], bins=50, alpha=0.7, label='s [m/s]', color='blue')
    ax3.hist(controls[:, 1], bins=50, alpha=0.7, label='omega [rad/s]', color='red')
    ax3.hist(controls[:, 2], bins=50, alpha=0.7, label='v_path [rad/s]', color='green')
    ax3.set_xlabel('Control Value', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Control Distributions', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Control vs state position
    ax4 = axes[1, 1]
    scatter = ax4.scatter(states[:, 0], states[:, 1], c=controls[:, 0], 
                         cmap='plasma', s=5, alpha=0.6)
    ax4.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Reference')
    ax4.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax4.set_title('Speed (s) vs Position', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    plt.colorbar(scatter, ax=ax4, label='Speed [m/s]')
    
    plt.suptitle('Dataset Verification', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/dataset_verification.png', dpi=200, bbox_inches='tight')
    
    print(f"      Visualization saved to 'outputs/dataset_verification.png'")
    
    print("\n" + "="*70)
    print(" Verification Complete!")
    print("="*70)


if __name__ == "__main__":
    verify_dataset()
