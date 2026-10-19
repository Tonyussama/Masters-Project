import numpy as np
from tqdm import tqdm

try:
    from .reference import find_projection_index
except ImportError:
    from src.utils.reference import find_projection_index


class MPCDataCollector:
    """Collect training data from the MPC controller using corridor experiment methodology."""
    
    def __init__(self, solver, t_param, x_ref_array, y_ref_array, theta_ref_array, N, nu, ts, v_ref):
        """
        Initialize data collector.
        
        Args:
            solver: MPC solver instance
            t_param: Path parameter array
            x_ref_array: Reference x positions
            y_ref_array: Reference y positions
            theta_ref_array: Reference heading angles
            N: Prediction horizon
            nu: Number of control inputs
            ts: Time step
            v_ref: Reference path velocity
        """
        self.solver = solver
        self.t_param = t_param
        self.x_ref_array = x_ref_array
        self.y_ref_array = y_ref_array
        self.theta_ref_array = theta_ref_array
        self.N = N
        self.nu = nu
        self.ts = ts
        self.v_ref = v_ref

    def find_projection_index(self, pos):
        """Find index of closest reference point."""
        return find_projection_index(pos, self.x_ref_array, self.y_ref_array)

    def get_reference_horizon(self, theta_p0):
        """Generate reference horizon from current theta_p.
        
        Args:
            theta_p0: Current path parameter
        
        Returns:
            x_ref: Reference x positions over horizon
            y_ref: Reference y positions over horizon
            theta_ref: Reference heading over horizon
        """
        theta_ref_horizon_p = theta_p0 + self.v_ref * np.arange(self.N) * self.ts
        
        # Interpolate to get x, y, theta references
        x_ref = np.interp(theta_ref_horizon_p, self.t_param, self.x_ref_array)
        y_ref = np.interp(theta_ref_horizon_p, self.t_param, self.y_ref_array)
        
        # Use unwrap to handle angle jumps in theta_ref_array
        theta_ref = np.interp(theta_ref_horizon_p, self.t_param, np.unwrap(self.theta_ref_array))
        theta_ref = (theta_ref + np.pi) % (2 * np.pi) - np.pi  # Wrap back to [-pi, pi]
        return x_ref, y_ref, theta_ref
    
    def extract_reference_features(self, state, x_ref, y_ref, theta_ref):
        """Extract features from reference horizon in robot's local frame.
        
        Args:
            state: Current geometric state [x, y, theta]
            x_ref: Reference x positions over horizon
            y_ref: Reference y positions over horizon
            theta_ref: Reference heading over horizon
        
        Returns:
            Feature vector for neural network input
        """
        x, y, theta = state
        
        # Transform reference points to robot frame
        cos_theta = np.cos(-theta)
        sin_theta = np.sin(-theta)
        
        dx = x_ref - x
        dy = y_ref - y
        
        x_ref_local = dx * cos_theta - dy * sin_theta
        y_ref_local = dx * sin_theta + dy * cos_theta
        theta_ref_local = theta_ref - theta
        
        # Normalize angles to [-pi, pi]
        theta_ref_local = np.arctan2(np.sin(theta_ref_local), np.cos(theta_ref_local))
        
        # Use every 5th reference point to reduce dimension
        stride = 5
        features = np.concatenate([
            x_ref_local[::stride],
            y_ref_local[::stride],
            theta_ref_local[::stride]
        ])
        
        return features
    
    def collect_data_corridor(self, num_anchors=1000, offset_distance=0.1):
        """
        Collect training data using corridor experiment methodology.
        
        Args:
            num_anchors: Number of anchor points along the path
            offset_distance: Distance d for the 3x3 grid offsets
        
        Returns:
            states: Array of geometric states
            references: Array of reference features
            controls: Array of control actions
        """
        states = []
        references = []
        controls = []
        
        print(f"Collecting training data using corridor experiment methodology...")
        print(f"  - Number of anchors: {num_anchors}")
        print(f"  - 3x3 grid per anchor: {num_anchors * 9} total seeds")
        print(f"  - Offset distance: {offset_distance} m")
        
        # Step 1: Anchor Point Sampling
        # Dense sampling of anchor points along the reference trajectory
        anchor_indices = np.linspace(0, len(self.x_ref_array) - 1, num_anchors, dtype=int)
        
        # Offsets for 3x3 grid: {-d, 0, +d} in both lateral and longitudinal directions
        offsets = [-offset_distance, 0, offset_distance]
        
        total_seeds = num_anchors * 9
        seed_count = 0
        
        for anchor_idx in tqdm(anchor_indices, desc="Processing anchors"):
            # Get anchor point information
            x_anchor = self.x_ref_array[anchor_idx]
            y_anchor = self.y_ref_array[anchor_idx]
            theta_anchor = self.theta_ref_array[anchor_idx]
            theta_p_anchor = self.t_param[anchor_idx]
            
            # Calculate local tangent and normal vectors
            cos_theta = np.cos(theta_anchor)
            sin_theta = np.sin(theta_anchor)
            
            # Tangent direction (along path)
            tangent = np.array([cos_theta, sin_theta])
            # Normal direction (perpendicular to path)
            normal = np.array([-sin_theta, cos_theta])
            
            # Step 2: Local 3x3 Seed Generation
            for lat_offset in offsets:  # Lateral (cross-track) offset
                for lon_offset in offsets:  # Longitudinal (along-track) offset
                    # Step 3: State Initialization
                    # Position with offsets
                    x = x_anchor + lat_offset * normal[0] + lon_offset * tangent[0]
                    y = y_anchor + lat_offset * normal[1] + lon_offset * tangent[1]
                    
                    # Heading aligned with local path tangent
                    theta = theta_anchor
                    
                    # Path parameter matches anchor
                    theta_p = theta_p_anchor
                    
                    # Form augmented state
                    aug_state = np.array([x, y, theta, theta_p])
                    
                    # Step 4: Expert MPC Querying & Labeling
                    # Get reference horizon
                    x_ref, y_ref, theta_ref = self.get_reference_horizon(aug_state[3])
                    
                    # Solve MPC
                    p_vector = np.concatenate([aug_state, x_ref, y_ref, theta_ref])
                    u_init = [0.01] * (self.nu * self.N)
                    
                    result = self.solver.run(p=p_vector, initial_guess=u_init)
                    
                    if result.exit_status == 'Converged':
                        # Store only the first control action
                        u_optimal = result.solution[0:self.nu]
                        
                        # Store the 3D geometric state and the 3D control
                        states.append(aug_state[:3])
                        ref_features = self.extract_reference_features(
                            aug_state[:3], x_ref, y_ref, theta_ref
                        )
                        references.append(ref_features)
                        controls.append(u_optimal)
                        seed_count += 1
        
        print(f"Successfully collected {len(states)} samples out of {total_seeds} seeds")
        print(f"Convergence rate: {seed_count/total_seeds*100:.1f}%")
        
        return np.array(states), np.array(references), np.array(controls)
    
    def collect_data(self, num_samples=8000, add_noise=True):
        """
        Collect training data by sampling random states and solving MPC (legacy method).
        
        Args:
            num_samples: Number of samples to collect
            add_noise: Whether to add noise to states
        
        Returns:
            states: Array of geometric states
            references: Array of reference features
            controls: Array of control actions
        """
        states = []
        references = []
        controls = []
        
        print(f"Collecting training data from MPC (legacy random sampling)...")
        
        for _ in tqdm(range(num_samples), desc="Progress"):
            # Random position along the path
            idx = np.random.randint(0, len(self.x_ref_array))
            
            # Add noise around reference trajectory
            if add_noise:
                x = self.x_ref_array[idx] + np.random.normal(0, 0.1)
                y = self.y_ref_array[idx] + np.random.normal(0, 0.1)
                theta = self.theta_ref_array[idx] + np.random.normal(0, 0.3)
            else:
                x = self.x_ref_array[idx]
                y = self.y_ref_array[idx]
                theta = self.theta_ref_array[idx]

            # Find corresponding path parameter
            theta_p = self.t_param[idx]
            current_aug_state = np.array([x, y, theta, theta_p])
            
            # Get reference horizon
            x_ref, y_ref, theta_ref = self.get_reference_horizon(current_aug_state[3])
            
            # Solve MPC
            p_vector = np.concatenate([current_aug_state, x_ref, y_ref, theta_ref])
            u_init = [0.01] * (self.nu * self.N)
            
            result = self.solver.run(p=p_vector, initial_guess=u_init)
            
            if result.exit_status == 'Converged':
                # Store only the first control action
                u_optimal = result.solution[0:self.nu]
                
                # Store the 3D geometric state and the 3D control
                states.append(current_aug_state[:3])
                ref_features = self.extract_reference_features(
                    current_aug_state[:3], x_ref, y_ref, theta_ref
                )
                references.append(ref_features)
                controls.append(u_optimal)
        
        print(f"Successfully collected {len(states)} samples")
        return np.array(states), np.array(references), np.array(controls)
