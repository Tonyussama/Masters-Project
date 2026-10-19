import numpy as np


def find_projection_index(pos, x_ref, y_ref):
    """Find the index of the reference point closest to current state.
    
    Args:
        pos: Current position [x, y]
        x_ref: Array of reference x positions
        y_ref: Array of reference y positions
    
    Returns:
        Index of closest reference point
    """
    distances = np.sqrt((x_ref - pos[0])**2 + (y_ref - pos[1])**2)
    return np.argmin(distances)


def generate_ellipse_reference(total_points=10000, height=4.0, width=0.2):
    """Generate elliptical reference trajectory.
    
    Args:
        total_points: Number of points in the trajectory
        height: Ellipse height
        width: Ellipse width
    
    Returns:
        t_param: Parameter array [0, 2*pi]
        x_ref_array: Reference x positions
        y_ref_array: Reference y positions
        theta_ref_array: Reference heading angles
    """
    t_param = np.linspace(0, 2 * np.pi, total_points)
    
    a = width / 2
    b = height / 2
    
    x_ref_array = a * np.cos(t_param)
    y_ref_array = b * np.sin(t_param)
    
    dx_ref = np.gradient(x_ref_array, t_param)
    dy_ref = np.gradient(y_ref_array, t_param)
    theta_ref_array = np.arctan2(dy_ref, dx_ref)
    
    return t_param, x_ref_array, y_ref_array, theta_ref_array
