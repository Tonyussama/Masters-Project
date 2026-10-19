import numpy as np
import casadi.casadi as cs


class MPCDirect:
    """MPC controller using direct CasADi optimization (no OpEn/Rust required)."""
    
    def __init__(self, nu=3, nx_aug=4, N=70, ts=0.03, 
                 q=100, qtheta=10, r_control=0.1, r_v=0.001, v_ref=0.2):
        """
        Initialize direct MPC controller.
        
        Args:
            nu: Number of control inputs [s, omega, v]
            nx_aug: Augmented state dimension [x, y, theta, theta_p]
            N: Prediction horizon
            ts: Time step
            q: Position error weight
            qtheta: Heading error weight
            r_control: Control effort weight for s and omega
            r_v: Control effort weight for v
            v_ref: Reference path velocity
        """
        self.nu = nu
        self.nx_aug = nx_aug
        self.N = N
        self.ts = ts
        self.q = q
        self.qtheta = qtheta
        self.r_control = r_control
        self.r_v = r_v
        self.v_ref = v_ref
        
        # Control bounds
        self.s_min, self.s_max = 0.02, 4.0
        self.w_min, self.w_max = -7.0, 7.0
        self.v_path_min, self.v_path_max = 0.01, 0.4
        
        # Build optimization problem
        self._build_optimizer()
    
    def _build_optimizer(self):
        """Build CasADi optimization problem."""
        # Decision variables: control sequence
        U = cs.SX.sym('U', self.nu * self.N)
        
        # Parameters: initial state + reference horizon
        P = cs.SX.sym('P', self.nx_aug + 3 * self.N)
        
        # Extract parameters
        x0 = P[0]
        y0 = P[1]
        theta0 = P[2]
        theta_p0 = P[3]
        
        x_ref_sym = P[self.nx_aug : self.nx_aug + self.N]
        y_ref_sym = P[self.nx_aug + self.N : self.nx_aug + 2 * self.N]
        theta_ref_sym = P[self.nx_aug + 2 * self.N : self.nx_aug + 3 * self.N]
        
        # Build cost function
        cost = 0
        x = x0
        y = y0
        theta = theta0
        theta_p = theta_p0
        
        for t_step in range(self.N):
            xref_t = x_ref_sym[t_step]
            yref_t = y_ref_sym[t_step]
            thetaref_t = theta_ref_sym[t_step]
            
            # Tracking cost
            cost += self.q * ((x - xref_t)**2 + (y - yref_t)**2)
            heading_error = cs.atan2(cs.sin(theta - thetaref_t), cs.cos(theta - thetaref_t))
            cost += self.qtheta * (heading_error)**2
            
            # Control cost
            s_k = U[t_step * self.nu]
            w = U[t_step * self.nu + 1]
            v_k = U[t_step * self.nu + 2]
            cost += self.r_control * (s_k**2 + w**2) + self.r_v * (v_k - self.v_ref)**2
            
            # Dynamics
            x = x + self.ts * s_k * cs.cos(theta)
            y = y + self.ts * s_k * cs.sin(theta)
            theta = theta + self.ts * w
            theta_p = theta_p + self.ts * v_k
        
        # Terminal cost
        xref_N = x_ref_sym[self.N - 1]
        yref_N = y_ref_sym[self.N - 1]
        thetaref_N = theta_ref_sym[self.N - 1]
        terminal_pos_error = (x - xref_N)**2 + (y - yref_N)**2
        terminal_heading_error = cs.atan2(cs.sin(theta - thetaref_N), cs.cos(theta - thetaref_N))
        cost += self.q * terminal_pos_error + self.qtheta * terminal_heading_error**2
        
        # Constraints
        constraints = []
        lbx = []
        ubx = []
        
        for t_step in range(self.N):
            s_k = U[t_step * self.nu]
            w = U[t_step * self.nu + 1]
            v_k = U[t_step * self.nu + 2]
            
            lbx.extend([self.s_min, self.w_min, self.v_path_min])
            ubx.extend([self.s_max, self.w_max, self.v_path_max])
        
        # Create NLP solver
        nlp = {'x': U, 'p': P, 'f': cost}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100}
        self.solver = cs.nlpsol('solver', 'ipopt', nlp, opts)
    
    def get_first_control(self, aug_state, x_ref, y_ref, theta_ref):
        """
        Get first control action using direct optimization.
        
        Args:
            aug_state: Augmented state [x, y, theta, theta_p]
            x_ref: Reference x positions over horizon
            y_ref: Reference y positions over horizon
            theta_ref: Reference heading over horizon
        
        Returns:
            First control action [s, omega, v]
        """
        p_vector = np.concatenate([aug_state, x_ref, y_ref, theta_ref])
        u_init = np.array([0.01] * (self.nu * self.N))
        
        # Solve
        try:
            result = self.solver(x0=u_init, p=p_vector, lbx=self.lb_input, ubx=self.ub_input)
            u_optimal = result['x'].full().flatten()
            return u_optimal[0:self.nu]
        except Exception as e:
            # Fallback to simple control if optimization fails
            print(f"      MPC solve error: {e}")
            return np.array([self.v_ref, 0.0, self.v_ref])
    
    @property
    def lb_input(self):
        """Lower bounds for control inputs."""
        lb = []
        for _ in range(self.N):
            lb.extend([self.s_min, self.w_min, self.v_path_min])
        return np.array(lb)
    
    @property
    def ub_input(self):
        """Upper bounds for control inputs."""
        ub = []
        for _ in range(self.N):
            ub.extend([self.s_max, self.w_max, self.v_path_max])
        return np.array(ub)
