import numpy as np
from scipy.linalg import solve_discrete_are, inv


class LQRController:
    """LQR controller with proper linearization and gain scheduling."""
    
    def __init__(self, v_ref=0.2, dt=0.03):
        """
        Initialize LQR controller.
        
        Args:
            v_ref: Reference forward velocity
            dt: Time step
        """
        self.v_ref = v_ref
        self.dt = dt
        
        # LQR weights
        self.Q = np.diag([100.0, 100.0, 50.0])  # State cost [e_along, e_cross, e_theta]
        self.R = np.diag([0.1, 0.1])         # Control cost [s, omega]
        
        # Control limits
        self.s_min, self.s_max = 0.01, 0.4
        self.omega_min, self.omega_max = -7.0, 7.0
        
        # Store reference path
        self.x_ref = None
        self.y_ref = None
        self.theta_ref = None
        self.t_param = None
        
        # Precomputed gains and reference controls
        self.gains = []
        self.v_ref_array = []
        self.omega_ref_array = []
        
        # For nearest-point search
        self.prev_idx = 0
    
    def set_reference(self, x_ref, y_ref, theta_ref, t_param):
        """Set reference path and precompute LQR gains."""
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.theta_ref = theta_ref
        self.t_param = t_param
        
        n = len(x_ref)
        self.gains = []
        self.v_ref_array = np.zeros(n)
        self.omega_ref_array = np.zeros(n)
        
        # Assume dθ_p/dt ≈ 0.2 rad/s
        dtheta_p_dt = 0.2
        
        for i in range(n):
            j = (i + 1) % n
            dx = x_ref[j] - x_ref[i]
            dy = y_ref[j] - y_ref[i]
            ds = np.hypot(dx, dy) + 1e-8
            dtheta = np.arctan2(np.sin(theta_ref[j] - theta_ref[i]), 
                               np.cos(theta_ref[j] - theta_ref[i]))
            
            # Reference controls
            v_r = ds * dtheta_p_dt
            omega_r = dtheta * dtheta_p_dt
            
            self.v_ref_array[i] = v_r
            self.omega_ref_array[i] = omega_r
            
            # Linearize around reference point
            # Error state: e = [e_along, e_cross, e_theta]
            # Dynamics: de/dt = [v_r - v, omega_r - omega, -omega]
            # Discrete: e[k+1] = A*e[k] + B*δu
            # where δu = [v - v_r, omega - omega_r]
            
            A = np.eye(3) + self.dt * np.array([
                [0, 0, 0],
                [0, 0, v_r],
                [0, 0, 0]
            ])
            
            B = self.dt * np.array([
                [-1, 0],
                [0, 0],
                [0, -1]
            ])
            
            # Solve DARE
            P = solve_discrete_are(A, B, self.Q, self.R)
            K = inv(self.R + B.T @ P @ B) @ B.T @ P @ A
            
            self.gains.append(K)
    
    def _find_nearest_point(self, x, y):
        """Find nearest point on reference path using windowed search."""
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
        Compute LQR control action.
        
        Args:
            aug_state: Augmented state [x, y, theta, theta_p]
        
        Returns:
            Control [s, omega, v]
        """
        x, y, theta, theta_p = aug_state
        
        # Find nearest point on reference
        idx = self._find_nearest_point(x, y)
        
        x_ref_pt = self.x_ref[idx]
        y_ref_pt = self.y_ref[idx]
        theta_ref_pt = self.theta_ref[idx]
        
        # Transform error to path frame (FIXED: swapped e_along and e_cross)
        cos_ref = np.cos(theta_ref_pt)
        sin_ref = np.sin(theta_ref_pt)
        
        dx = x - x_ref_pt
        dy = y - y_ref_pt
        
        e_along = dx * cos_ref + dy * sin_ref  # Along-track error
        e_cross = -dx * sin_ref + dy * cos_ref  # Cross-track error
        e_theta = theta - theta_ref_pt
        e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))
        
        # LQR control
        error = np.array([e_along, e_cross, e_theta])
        K = self.gains[idx]
        delta_u = -K @ error
        
        # Reference controls
        v_r = self.v_ref_array[idx]
        omega_r = self.omega_ref_array[idx]
        
        # Total control
        s = v_r + delta_u[0]
        omega = omega_r + delta_u[1]
        
        # Apply constraints
        s = np.clip(s, self.s_min, self.s_max)
        omega = np.clip(omega, self.omega_min, self.omega_max)
        
        return np.array([s, omega, v_r])
