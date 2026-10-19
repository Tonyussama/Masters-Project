"""
Enhanced visualization of the reference path with multiple views,
robot poses, and curvature information.
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


def plot_enhanced_reference():
    """Generate comprehensive visualizations of the reference path."""
    # Generate reference trajectory
    total_points = 10000
    t_param, x_ref, y_ref, theta_ref = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    
    # Calculate curvature
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # =========================================
    # Plot 1: Full path with robot poses
    # =========================================
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x_ref, y_ref, 'b-', linewidth=2, label='Reference Path', alpha=0.5)
    
    # Plot robot poses at intervals
    pose_stride = 400
    for i in range(0, len(x_ref), pose_stride):
        x, y, theta = x_ref[i], y_ref[i], theta_ref[i]
        # Robot body (simple representation)
        robot_len = 0.08
        robot_width = 0.04
        
        # Robot corners in local frame
        corners_x = np.array([-robot_len/2, robot_len/2, robot_len/2, -robot_len/2])
        corners_y = np.array([-robot_width/2, -robot_width/2, robot_width/2, robot_width/2])
        
        # Transform to global frame
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        corners_global_x = x + corners_x * cos_t - corners_y * sin_t
        corners_global_y = y + corners_x * sin_t + corners_y * cos_t
        
        # Draw robot
        poly = plt.Polygon(np.column_stack([corners_global_x, corners_global_y]), 
                          facecolor='red', edgecolor='black', alpha=0.7)
        ax1.add_patch(poly)
        
        # Draw heading arrow
        arrow_len = 0.1
        ax1.arrow(x, y, arrow_len * cos_t, arrow_len * sin_t,
                 head_width=0.02, head_length=0.03, fc='green', ec='green')
    
    ax1.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax1.set_title('Full Path with Robot Poses', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # =========================================
    # Plot 2: Zoomed top section
    # =========================================
    ax2 = plt.subplot(2, 3, 2)
    mask = y_ref > 1.5
    ax2.plot(x_ref[mask], y_ref[mask], 'b-', linewidth=2.5, alpha=0.7)
    
    # Dense heading arrows
    for i in range(len(x_ref)):
        if mask[i] and i % 50 == 0:
            x, y, theta = x_ref[i], y_ref[i], theta_ref[i]
            ax2.arrow(x, y, 0.05 * np.cos(theta), 0.05 * np.sin(theta),
                     head_width=0.015, head_length=0.02, fc='red', ec='red', alpha=0.6)
    
    ax2.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax2.set_title('Top Section (High Curvature)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # =========================================
    # Plot 3: Zoomed bottom section
    # =========================================
    ax3 = plt.subplot(2, 3, 3)
    mask = y_ref < -1.5
    ax3.plot(x_ref[mask], y_ref[mask], 'b-', linewidth=2.5, alpha=0.7)
    
    for i in range(len(x_ref)):
        if mask[i] and i % 50 == 0:
            x, y, theta = x_ref[i], y_ref[i], theta_ref[i]
            ax3.arrow(x, y, 0.05 * np.cos(theta), 0.05 * np.sin(theta),
                     head_width=0.015, head_length=0.02, fc='red', ec='red', alpha=0.6)
    
    ax3.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax3.set_title('Bottom Section (High Curvature)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # =========================================
    # Plot 4: Curvature along path
    # =========================================
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(t_param, curvature, 'g-', linewidth=2)
    ax4.set_xlabel('Path Parameter θ_p [rad]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Curvature [1/m]', fontsize=11, fontweight='bold')
    ax4.set_title('Curvature Along Path', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 2*np.pi)
    
    # =========================================
    # Plot 5: Heading angle along path
    # =========================================
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t_param, theta_ref, 'm-', linewidth=2)
    ax5.set_xlabel('Path Parameter θ_p [rad]', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Heading θ [rad]', fontsize=11, fontweight='bold')
    ax5.set_title('Heading Angle Along Path', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 2*np.pi)
    
    # =========================================
    # Plot 6: Path parameter vs arc length
    # =========================================
    ax6 = plt.subplot(2, 3, 6)
    # Calculate arc length
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    ds = np.sqrt(dx**2 + dy**2)
    arc_length = np.cumsum(ds)
    arc_length = arc_length - arc_length[0]  # Start from 0
    
    ax6.plot(t_param, arc_length, 'c-', linewidth=2)
    ax6.set_xlabel('Path Parameter θ_p [rad]', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Arc Length [m]', fontsize=11, fontweight='bold')
    ax6.set_title('Arc Length vs Path Parameter', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 2*np.pi)
    
    plt.suptitle('Comprehensive Reference Path Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/reference_path_enhanced.png', dpi=200, bbox_inches='tight')
    # plt.show()  # Disabled for non-interactive backend
    
    print(f"Enhanced reference path visualization saved to 'outputs/reference_path_enhanced.png'")
    print(f"Total path length: {arc_length[-1]:.2f} m")
    print(f"Max curvature: {np.max(curvature):.2f} 1/m")
    print(f"Min curvature: {np.min(curvature):.2f} 1/m")


if __name__ == "__main__":
    plot_enhanced_reference()
