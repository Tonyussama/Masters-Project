"""
Visualize the reference elliptical path used for MPC training.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.reference import generate_ellipse_reference


def plot_reference_path():
    """Generate and plot the reference elliptical path."""
    # Generate reference trajectory
    total_points = 10000
    t_param, x_ref, y_ref, theta_ref = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the full path
    ax.plot(x_ref, y_ref, 'b-', linewidth=2, label='Reference Path', alpha=0.7)
    
    # Plot heading arrows at intervals
    stride = 200  # Plot arrow every 200 points
    for i in range(0, len(x_ref), stride):
        x, y, theta = x_ref[i], y_ref[i], theta_ref[i]
        # Arrow length
        arrow_len = 0.15
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.03, head_length=0.05, 
                fc='red', ec='red', alpha=0.6)
    
    # Mark start and end points
    ax.plot(x_ref[0], y_ref[0], 'go', markersize=12, label='Start', markeredgecolor='black')
    ax.plot(x_ref[-1], y_ref[-1], 'rs', markersize=12, label='End', markeredgecolor='black')
    
    # Formatting
    ax.set_xlabel('x [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('y [m]', fontsize=12, fontweight='bold')
    ax.set_title('Reference Elliptical Path with Heading', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add path info
    info_text = f'Path Parameters:\n'
    info_text += f'  Height: {4.0} m\n'
    info_text += f'  Width: {0.2} m\n'
    info_text += f'  Points: {total_points}\n'
    info_text += f'  Path parameter: [0, 2π]'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('outputs/reference_path.png', dpi=200, bbox_inches='tight')
    # plt.show()  # Disabled for non-interactive backend
    
    print(f"Reference path visualization saved to 'outputs/reference_path.png'")
    print(f"Path dimensions: {0.2}m width × {4.0}m height")


if __name__ == "__main__":
    plot_reference_path()
