import numpy as np
import casadi.casadi as cs


class KinematicUnicycle:
    """Kinematic unicycle model for path following.
    
    State: [x, y, theta]
    Control: [v, omega] (linear velocity, angular velocity)
    """
    
    def __init__(self, dt=0.03, initial_state=None):
        """
        Initialize the unicycle model.
        
        Args:
            dt: Time step in seconds
            initial_state: Initial state [x, y, theta]. Defaults to [0.1, 0.0, pi/2]
        """
        self.dt = dt
        if initial_state is None:
            self.state = np.array([0.1, 0.0, np.pi/2])
        else:
            self.state = np.array(initial_state)
    
    def get_state(self):
        """Get current state."""
        return self.state.copy()
    
    def reset(self, initial_state=None):
        """Reset to initial state."""
        if initial_state is None:
            self.state = np.array([0.1, 0.0, np.pi/2])
        else:
            self.state = np.array(initial_state)
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def step(self, u):
        """
        Apply control input and advance one time step.
        
        Args:
            u: Control input [v, omega] (linear velocity, angular velocity)
        
        Returns:
            New state [x, y, theta]
        """
        v, omega = u[0], u[1]
        
        # Kinematic equations
        x_new = self.state[0] + self.dt * v * np.cos(self.state[2])
        y_new = self.state[1] + self.dt * v * np.sin(self.state[2])
        theta_new = self.state[2] + self.dt * omega
        
        # Normalize angle
        theta_new = self._normalize_angle(theta_new)
        
        self.state = np.array([x_new, y_new, theta_new])
        return self.state.copy()
    
    @staticmethod
    def dynamics_symbolic(x, y, theta, v, omega, dt):
        """
        CasADi symbolic dynamics for use in optimization.
        
        Args:
            x, y, theta: CasADi symbols for state
            v, omega: CasADi symbols for control
            dt: Time step
        
        Returns:
            Next state as CasADi expressions
        """
        x_next = x + dt * v * cs.cos(theta)
        y_next = y + dt * v * cs.sin(theta)
        theta_next = theta + dt * omega
        
        # Normalize angle to [-pi, pi]
        theta_next = cs.atan2(cs.sin(theta_next), cs.cos(theta_next))
        
        return x_next, y_next, theta_next
