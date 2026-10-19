import numpy as np
from scipy.linalg import solve_discrete_are, inv


class LQRController:
    """Linear Quadratic Regulator with gain scheduling for path following."""
    
    def __init__(self, v_ref=0.2, dt=0.03):
        """
        Initialize LQR controller with gain scheduling.
        
        Args:
            v_ref: Reference forward velocity
            dt: Time step
        """
        self.v_ref = v_ref
        self.dt = dt
        
        # LQR weights (tuned for tight tracking)
        self.Q = np.diag([5.0, 0.5, 10.0])  # State cost [e_lat, e_lon, e_theta]
        self.R = np.diag([0.3, 0.3])         # Control cost [v, omega]
        
        # Pre-compute gains for different curvatures (gain scheduling)
        self.curvature_bins = np.linspace(-5, 5, 21)  # 21 bins from -5 to 5
        self.gain_schedule = {}
        
        # Compute LQR gains for each curvature bin
        for k in self.curvature_bins:
            A = np.array([
                [1, self.v_ref * k * self.dt, -self.v_ref * self.dt],
                [-self.v_ref * k * self.dt, 1, 0],
                [0, 0, 1]
            ])
            B = np.array([
                [self.dt, 0],
                [0, self.v_ref * self.dt],
                [0, self.dt]
            ])
            
            P = solve_discrete_are(A, B, self.Q, self.R)
            K = inv(self.R + B.T @ P @ B) @ B.T @ P @ A
            self.gain_schedule[k] = K
        
        # Control limits
        self.v_min, self.v_max = 0.01, 0.4
        self.omega_min, self.omega_max = -4.0, 4.0
    
    def _get_gain(self, curvature):
        """Get LQR gain for given curvature using interpolation."""
        # Clamp curvature to bin range
        k_clamped = np.clip(curvature, self.curvature_bins[0], self.curvature_bins[-1])
        
        # Find nearest bins
        idx = np.searchsorted(self.curvature_bins, k_clamped)
        if idx == 0:
            return self.gain_schedule[self.curvature_bins[0]]
        elif idx >= len(self.curvature_bins):
            return self.gain_schedule[self.curvature_bins[-1]]
        else:
            # Linear interpolation
            k1, k2 = self.curvature_bins[idx-1], self.curvature_bins[idx]
            K1, K2 = self.gain_schedule[k1], self.gain_schedule[k2]
            alpha = (k_clamped - k1) / (k2 - k1) if k2 != k1 else 0.5
            return (1 - alpha) * K1 + alpha * K2
    
    def get_control(self, state, path_state):
        """
        Compute LQR control action with gain scheduling.
        
        Args:
            state: Current robot state [x, y, theta]
            path_state: Reference path state [x_ref, y_ref, theta_ref, k_ref]
        
        Returns:
            Control [v, omega]
        """
        x, y, theta = state
        x_ref, y_ref, theta_ref, k_ref = path_state
        
        # Transform error to path frame
        cos_ref = np.cos(theta_ref)
        sin_ref = np.sin(theta_ref)
        
        e_lat = (x - x_ref) * cos_ref + (y - y_ref) * sin_ref
        e_lon = -(x - x_ref) * sin_ref + (y - y_ref) * cos_ref
        e_theta = theta - theta_ref
        
        # Normalize angle error
        e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))
        
        # Get gain for current curvature
        K = self._get_gain(k_ref)
        
        # LQR control
        error = np.array([e_lat, e_lon, e_theta])
        u = -K @ error
        
        # Feedforward terms
        v_ff = self.v_ref
        omega_ff = self.v_ref * k_ref
        
        # Total control
        v = v_ff + u[0]
        omega = omega_ff + u[1]
        
        # Apply constraints
        v = np.clip(v, self.v_min, self.v_max)
        omega = np.clip(omega, self.omega_min, self.omega_max)
        
        return np.array([v, omega])
