import opengen as og
import casadi.casadi as cs
import numpy as np
import sys
from pathlib import Path


class MPCController:
    """Model Predictive Controller for path following with augmented state."""
    
    def __init__(self, nu=3, nx_aug=4, N=70, ts=0.03, 
                 q=100, qtheta=10, r_control=0.1, r_v=0.001, v_ref=0.2):
        """
        Initialize MPC controller.
        
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
        
        self.solver = None
        self.built = False
    
    def build_optimizer(self, build_dir="my_optimizers"):
        """Build the optimization problem using OpEn."""
        # Check if already built
        try:
            sys.path.insert(0, f"{build_dir}/path_following_aug")
            import path_following_aug
            self.solver = path_following_aug.solver()
            self.built = True
            return self.solver
        except:
            pass  # Not built yet, proceed with building
        
        # Decision variables: control sequence
        u = cs.SX.sym('u', self.nu * self.N)
        
        # Parameters: initial state + reference horizon
        p = cs.SX.sym('p', self.nx_aug + 3 * self.N)
        
        # Extract parameters
        x0 = p[0]
        y0 = p[1]
        theta0 = p[2]
        theta_p0 = p[3]
        
        x_ref_sym = p[self.nx_aug : self.nx_aug + self.N]
        y_ref_sym = p[self.nx_aug + self.N : self.nx_aug + 2 * self.N]
        theta_ref_sym = p[self.nx_aug + 2 * self.N : self.nx_aug + 3 * self.N]
        
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
            s_k = u[t_step * self.nu]
            w = u[t_step * self.nu + 1]
            v_k = u[t_step * self.nu + 2]
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
        umin = []
        umax = []
        for _ in range(self.N):
            umin.extend([self.s_min, self.w_min, self.v_path_min])
            umax.extend([self.s_max, self.w_max, self.v_path_max])
        
        bounds = og.constraints.Rectangle(umin, umax)
        
        # Build optimizer
        problem = og.builder.Problem(u, p, cost).with_constraints(bounds)
        
        build_config = og.config.BuildConfiguration() \
            .with_build_directory(build_dir) \
            .with_build_mode("release") \
            .with_tcp_interface_config() \
            .with_build_python_bindings()
        
        meta = og.config.OptimizerMeta().with_optimizer_name("path_following_aug")
        solver_config = og.config.SolverConfiguration().with_tolerance(1e-5)
        
        builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config)
        builder.build()
        
        # Import solver
        sys.path.insert(0, f"{build_dir}/path_following_aug")
        import path_following_aug
        self.solver = path_following_aug.solver()
        self.built = True
        
        return self.solver
    
    def solve(self, aug_state, x_ref, y_ref, theta_ref):
        """
        Solve MPC problem for current state.
        
        Args:
            aug_state: Augmented state [x, y, theta, theta_p]
            x_ref: Reference x positions over horizon
            y_ref: Reference y positions over horizon
            theta_ref: Reference theta over horizon
        
        Returns:
            Optimal control sequence [s, omega, v] * N
        """
        if not self.built:
            raise RuntimeError("MPC optimizer not built. Call build_optimizer() first.")
        
        p_vector = np.concatenate([aug_state, x_ref, y_ref, theta_ref])
        u_init = [0.01] * (self.nu * self.N)
        
        result = self.solver.run(p=p_vector, initial_guess=u_init)
        
        if result.exit_status == 'Converged':
            return result.solution
        else:
            raise RuntimeError(f"MPC solver failed: {result.exit_status}")
    
    def get_first_control(self, aug_state, x_ref, y_ref, theta_ref, u_init=None):
        """Get only the first control action, with optional warm start and timing."""
        if u_init is None:
            u_init = [0.01] * (self.nu * self.N)
        
        u_sequence = self.solve(aug_state, x_ref, y_ref, theta_ref, u_init=u_init)
        return u_sequence[0:self.nu], self.last_solve_time_ms
