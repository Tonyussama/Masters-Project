import numpy as np


class StanleyController:
    """Stanley controller for path following."""
    
    def __init__(self, v_ref=0.2, dt=0.03, k_gain=2.5, k_soft=0.1):
        """
        Initialize Stanley controller.
        
        Args:
            v_ref: Reference forward velocity
            dt: Time step
            k_gain: Cross-track error gain
            k_soft: Softening constant to prevent division by zero
        """
        self.v_ref = v_ref
        self.dt = dt
        self.k_gain = k_gain
        self.k_soft = k_soft
        
        # Control limits
        self.v_min, self.v_max = 0.01, 0.4
        self.omega_min, self.omega_max = -4.0, 4.0
    
    def get_control(self, state, path_state):
        """
        Compute Stanley control action.
        
        Args:
            state: Current robot state [x, y, theta]
            path_state: Reference path state [x_ref, y_ref, theta_ref, k_ref]
        
        Returns:
            Control [v, omega]
        """
        x, y, theta = state
        x_ref, y_ref, theta_ref, k_ref = path_state
        
        # Cross-track error (signed distance to path)
        # Positive if robot is to the left of the path
        e_ct = -(x - x_ref) * np.sin(theta_ref) + (y - y_ref) * np.cos(theta_ref)
        
        # Heading error
        e_heading = theta_ref - theta
        e_heading = np.arctan2(np.sin(e_heading), np.cos(e_heading))
        
        # Stanley steering angle
        delta = e_heading + np.arctan2(self.k_gain * e_ct, self.v_ref + self.k_soft)
        
        # For unicycle, use angular velocity directly (no steering angle conversion)
        # The Stanley law gives desired heading change rate
        omega = delta * self.v_ref / 0.1  # Divide by characteristic length
        
        # Add feedforward for path curvature
        omega += self.v_ref * k_ref
        
        # Velocity control (simple proportional to reduce speed when turning)
        v = self.v_ref * (1.0 - 0.3 * abs(delta))
        
        # Apply constraints
        v = np.clip(v, self.v_min, self.v_max)
        omega = np.clip(omega, self.omega_min, self.omega_max)
        
        return np.array([v, omega])
