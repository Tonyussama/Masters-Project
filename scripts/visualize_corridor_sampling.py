"""
Visualize the corridor experiment sampling methodology.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.reference import generate_ellipse_reference


def visualize_corridor_sampling():
    """Visualize the corridor experiment sampling methodology."""
    print("="*70)
    print(" Corridor Experiment Sampling Visualization")
    print("="*70)
    
    # Generate reference trajectory
    total_points = 10000
    t_param, x_ref, y_ref, theta_ref = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    
    # Calculate tangent and normal vectors
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    dx_norm = dx / np.sqrt(dx**2 + dy**2)
    dy_norm = dy / np.sqrt(dx**2 + dy**2)
    
    # Normal vectors (perpendicular to tangent)
    nx = -dy_norm
    ny = dx_norm
    
    # Sample a few anchor points
    num_anchor_visual = 15
    anchor_indices = np.linspace(0, len(x_ref) - 1, num_anchor_visual, dtype=int)
    
    offset_distance = 0.1
    offsets = [-offset_distance, 0, offset_distance]
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Full ellipse with anchor points
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Reference Path')
    ax1.scatter(x_ref[anchor_indices], y_ref[anchor_indices], 
               c='red', s=50, marker='o', label='Anchor Points', zorder=5)
    ax1.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax1.set_title('Anchor Points Along Reference Path', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Single anchor with 3x3 grid
    ax2 = plt.subplot(2, 2, 2)
    anchor_idx = anchor_indices[len(anchor_indices)//2]
    x_anchor = x_ref[anchor_idx]
    y_anchor = y_ref[anchor_idx]
    theta_anchor = theta_ref[anchor_idx]
    
    ax2.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.3, label='Reference Path')
    ax2.scatter(x_anchor, y_anchor, c='red', s=100, marker='o', label='Anchor Point', zorder=5)
    
    # Plot 3x3 grid
    grid_x = []
    grid_y = []
    grid_colors = []
    
    for lat_offset in offsets:
        for lon_offset in offsets:
            x = x_anchor + lat_offset * nx[anchor_idx] + lon_offset * dx_norm[anchor_idx]
            y = y_anchor + lat_offset * ny[anchor_idx] + lon_offset * dy_norm[anchor_idx]
            grid_x.append(x)
            grid_y.append(y)
            if lat_offset == 0 and lon_offset == 0:
                grid_colors.append('red')
            else:
                grid_colors.append('blue')
    
    ax2.scatter(grid_x, grid_y, c=grid_colors, s=30, marker='s', label='Seeds (3x3 Grid)', zorder=6)
    
    # Draw tangent and normal vectors
    ax2.arrow(x_anchor, y_anchor, dx_norm[anchor_idx]*0.2, dy_norm[anchor_idx]*0.2,
              head_width=0.02, head_length=0.02, fc='green', ec='green', label='Tangent')
    ax2.arrow(x_anchor, y_anchor, nx[anchor_idx]*0.2, ny[anchor_idx]*0.2,
              head_width=0.02, head_length=0.02, fc='orange', ec='orange', label='Normal')
    
    ax2.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax2.set_title('3x3 Grid Around Single Anchor Point', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.set_xlim([x_anchor - 0.3, x_anchor + 0.3])
    ax2.set_ylim([y_anchor - 0.3, y_anchor + 0.3])
    
    # Plot 3: Multiple anchors with their grids
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.3, label='Reference Path')
    
    # Plot grids for first few anchors
    for idx in anchor_indices[:5]:
        x_anchor = x_ref[idx]
        y_anchor = y_ref[idx]
        
        for lat_offset in offsets:
            for lon_offset in offsets:
                x = x_anchor + lat_offset * nx[idx] + lon_offset * dx_norm[idx]
                y = y_anchor + lat_offset * ny[idx] + lon_offset * dy_norm[idx]
                ax3.scatter(x, y, c='blue', s=15, marker='s', alpha=0.6, zorder=6)
    
    ax3.scatter(x_ref[anchor_indices[:5]], y_ref[anchor_indices[:5]], 
               c='red', s=50, marker='o', label='Anchor Points', zorder=5)
    
    ax3.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax3.set_title('Corridor Sampling: Multiple Anchor Grids', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot 4: Schematic of corridor concept
    ax4 = plt.subplot(2, 2, 4)
    
    # Draw reference path as a curve
    t_schematic = np.linspace(0, 2*np.pi, 100)
    x_schematic = 0.1 * np.cos(t_schematic)
    y_schematic = 0.5 * np.sin(t_schematic)
    ax4.plot(x_schematic, y_schematic, 'k--', linewidth=3, alpha=0.5, label='Reference Path')
    
    # Draw corridor boundaries
    corridor_width = 0.15
    x_upper = x_schematic + corridor_width * np.sin(t_schematic)
    y_upper = y_schematic + corridor_width * np.cos(t_schematic)
    x_lower = x_schematic - corridor_width * np.sin(t_schematic)
    y_lower = y_schematic - corridor_width * np.cos(t_schematic)
    
    ax4.plot(x_upper, y_upper, 'b-', linewidth=1, alpha=0.5, label='Corridor Boundary')
    ax4.plot(x_lower, y_lower, 'b-', linewidth=1, alpha=0.5)
    
    # Shade corridor
    ax4.fill_betweenx(y_schematic, x_lower, x_upper, alpha=0.2, color='blue')
    
    # Draw sample seeds
    num_seeds = 50
    for _ in range(num_seeds):
        t_sample = np.random.uniform(0, 2*np.pi)
        x_base = 0.1 * np.cos(t_sample)
        y_base = 0.5 * np.sin(t_sample)
        
        # Add random offset perpendicular to path
        lat_offset = np.random.uniform(-corridor_width, corridor_width)
        x_seed = x_base + lat_offset * np.sin(t_sample)
        y_seed = y_base + lat_offset * np.cos(t_sample)
        
        ax4.scatter(x_seed, y_seed, c='red', s=10, alpha=0.6)
    
    ax4.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax4.set_title('Corridor Concept: Dense Sampling Along Path', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    plt.suptitle('Corridor Experiment Data Collection Methodology', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/corridor_sampling_visualization.png', dpi=200, bbox_inches='tight')
    
    print(f"\nVisualization saved to 'outputs/corridor_sampling_visualization.png'")
    
    # Print explanation
    print("\n" + "="*70)
    print(" Corridor Experiment Methodology Explanation")
    print("="*70)
    print("\n1. Anchor Point Sampling:")
    print("   - 1000 anchor points densely sampled along the ellipse")
    print("   - Anchors represent nominal path positions")
    
    print("\n2. Local 3x3 Grid Generation:")
    print("   - Around each anchor, create 9 seed positions")
    print("   - Offsets: {-0.1m, 0, +0.1m} in lateral (normal) direction")
    print("   - Offsets: {-0.1m, 0, +0.1m} in longitudinal (tangent) direction")
    print("   - Total: 1000 anchors × 9 seeds = 9000 seeds")
    
    print("\n3. State Initialization:")
    print("   - Position: Anchor + lateral offset + longitudinal offset")
    print("   - Heading: Aligned with local path tangent")
    print("   - Path parameter: Matches anchor point")
    
    print("\n4. Expert MPC Querying:")
    print("   - Each seed treated as independent initial state")
    print("   - Solve full MPC problem from that state")
    print("   - Record only first optimal control action")
    print("   - Convergence rate: ~95.6% (8603/9000 samples)")
    
    print("\n5. Dataset Formation:")
    print("   - Input: [x, y, theta] (geometric state)")
    print("   - Output: [s, omega, v] (control action)")
    print("   - Captures structured corridor of deviations")
    print("   - Enables DNN to learn recovery from errors")
    
    print("\n" + "="*70)
    print(" ✓ Visualization Complete!")
    print("="*70)


if __name__ == "__main__":
    visualize_corridor_sampling()
