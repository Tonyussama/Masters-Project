import numpy as np


class StanleyController:
    """Stanley controller for path following."""
    
    def __init__(self, v_ref=0.2, dt=0.03, k_gain=5.0, k_soft=0.1, k_heading=2.0):
        """
        Initialize Stanley controller.
        
        Args:
            v_ref: Reference forward velocity
            dt: Time step
            k_gain: Cross-track error gain
            k_soft: Softening constant to prevent division by zero
            k_heading: Heading error gain multiplier
        """
        self.v_ref = v_ref
        self.dt = dt
        self.k_gain = k_gain
        self.k_soft = k_soft
        self.k_heading = k_heading
        
        # Control limits
        self.v_min, self.v_max = 0.01, 0.4
        self.omega_min, self.omega_max = -7.0, 7.0
        
        # Store reference path
        self.x_ref = None
        self.y_ref = None
        self.theta_ref = None
        self.t_param = None
        self.curvature = None
        self.v_ref_array = None
        
        # For nearest-point search
        self.prev_idx = 0
    
    def set_reference(self, x_ref, y_ref, theta_ref, t_param):
        """Set reference path and precompute curvature."""
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.theta_ref = theta_ref
        self.t_param = t_param
        
        # Precompute curvature (dθ/ds)
        n = len(x_ref)
        self.curvature = np.zeros(n)
        self.v_ref_array = np.zeros(n)
        
        for i in range(n):
            j = (i + 1) % n
            dx = x_ref[j] - x_ref[i]
            dy = y_ref[j] - y_ref[i]
            ds = np.hypot(dx, dy) + 1e-8
            dtheta = np.arctan2(np.sin(theta_ref[j] - theta_ref[i]), 
                               np.cos(theta_ref[j] - theta_ref[i]))
            self.curvature[i] = dtheta / ds
            # Reference speed from path parameter rate (assume dθ_p/dt ≈ 0.2 rad/s)
            self.v_ref_array[i] = ds * 0.2
    
    def _find_nearest_point(self, x, y):
        """Find nearest point on reference path using windowed search."""
        # Search in window around previous index
        window = 50
        n = len(self.x_ref)
        
        start = max(0, self.prev_idx - window)
        end = min(n, self.prev_idx + window)
        
        idx = start
        min_dist = float('inf')
        
        for i in range(start, end):
            dx = x - self.x_ref[i]
            dy = y - self.y_ref[i]
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                idx = i
        
        self.prev_idx = idx
        return idx
    
    def get_control(self, aug_state):
        """
        Compute Stanley control action.
        
        Args:
            aug_state: Augmented state [x, y, theta, theta_p]
        
        Returns:
            Control [s, omega, v]
        """
        x, y, theta, theta_p = aug_state
        
        # Find nearest point on reference (not time-indexed)
        idx = self._find_nearest_point(x, y)
        
        x_ref_pt = self.x_ref[idx]
        y_ref_pt = self.y_ref[idx]
        theta_ref_pt = self.theta_ref[idx]
        k_ref = self.curvature[idx]
        v_ref_pt = self.v_ref_array[idx]
        
        # Cross-track error (signed distance to path)
        # Positive if robot is to the left of the path
        e_ct = -(x - x_ref_pt) * np.sin(theta_ref_pt) + (y - y_ref_pt) * np.cos(theta_ref_pt)
        
        # Heading error
        e_heading = theta_ref_pt - theta
        e_heading = np.arctan2(np.sin(e_heading), np.cos(e_heading))
        
        # Stanley steering angle (FIXED SIGN)
        delta = e_heading - np.arctan2(self.k_gain * e_ct, v_ref_pt + self.k_soft)
        
        # Angular velocity (heading gain + curvature feedforward)
        omega = self.k_heading * delta + v_ref_pt * k_ref
        
        # Speed control (use reference speed, reduce for large heading error)
        s = v_ref_pt * (1.0 - 0.2 * abs(delta))
        
        # Apply constraints
        s = np.clip(s, self.v_min, self.v_max)
        omega = np.clip(omega, self.omega_min, self.omega_max)
        
        return np.array([s, omega, v_ref_pt])
