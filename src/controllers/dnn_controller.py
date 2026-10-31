import numpy as np
import torch
import torch.nn as nn


class MPCNeuralNetwork(nn.Module):
    """Deep Neural Network that mimics the MPC controller.
    
    Note: This version uses only robot-frame reference features as input,
    since the path preview already encodes the robot's pose relative to the path.
    This makes the DNN generalizable to any reference path.
    """
    
    def __init__(self, ref_feature_dim=42, hidden_dims=[1024, 1024, 512, 256, 128], output_dim=2):
        super(MPCNeuralNetwork, self).__init__()

        # Input normalization for reference features only
        self.register_buffer('ref_mean', torch.zeros(ref_feature_dim))
        self.register_buffer('ref_std', torch.ones(ref_feature_dim))
        
        input_dim = ref_feature_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))  # Output: [s, omega] (dropped v_path)
        
        self.network = nn.Sequential(*layers)
        
        # Output constraints matching augmented MPC
        self.s_min, self.s_max = 0.02, 4.0
        self.w_min, self.w_max = -7.0, 7.0

    def fit_normalization(self, refs):
        """Compute normalization statistics from training data"""
        self.ref_mean = torch.FloatTensor(refs.mean(axis=0))
        self.ref_std = torch.FloatTensor(refs.std(axis=0) + 1e-8)

    def forward(self, ref_features):
        # Normalize inputs
        ref_features = (ref_features - self.ref_mean) / self.ref_std
        
        output = self.network(ref_features)
        
        # Scale outputs to match augmented MPC constraints
        s = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(output[:, 0:1])
        omega = self.w_max * torch.tanh(output[:, 1:2])
        
        return torch.cat([s, omega], dim=-1)


class DNNController:
    """Wrapper for using the trained DNN as a controller with MPC fallback."""
    
    def __init__(self, model, collector, t_param, v_ref=0.2, mpc_solver=None, 
                 fallback_threshold=0.3, cross_track_threshold=0.5):
        """
        Initialize DNN controller.
        
        Args:
            model: Trained neural network model
            collector: MPCDataCollector instance for reference generation
            t_param: Path parameter array
            v_ref: Reference path velocity (default: 0.2)
            mpc_solver: Optional MPC solver for fallback (None = no fallback)
            fallback_threshold: Cross-track error threshold for MPC fallback (meters)
            cross_track_threshold: Maximum allowed cross-track error before triggering fallback
        """
        self.model = model
        self.collector = collector
        self.t_param = t_param
        self.v_ref = v_ref
        self.mpc_solver = mpc_solver
        self.fallback_threshold = fallback_threshold
        self.cross_track_threshold = cross_track_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.fallback_count = 0
        self.total_calls = 0
    
    def get_control(self, aug_state):
        """
        Get control action from DNN given current state.
        
        Args:
            aug_state: Augmented state [x, y, theta, theta_p]
        
        Returns:
            Control action [s, omega, v] (v is set to v_ref constant)
        """
        self.total_calls += 1
        
        # The DNN uses only robot-frame reference features
        geometric_state = aug_state[:3]
        
        # Reference features are based on the geometric state
        x_ref, y_ref, theta_ref = self.collector.get_reference_horizon(aug_state[3])
        ref_features = self.collector.extract_reference_features(geometric_state, x_ref, y_ref, theta_ref)
        
        # Check if we should use MPC fallback
        use_fallback = False
        
        if self.mpc_solver is not None:
            # Calculate cross-track error (distance to reference path)
            idx = self.collector.find_projection_index(geometric_state[:2])
            x_path = self.collector.x_ref_array[idx]
            y_path = self.collector.y_ref_array[idx]
            cross_track_error = np.sqrt((geometric_state[0] - x_path)**2 + (geometric_state[1] - y_path)**2)
            
            # Check if cross-track error exceeds threshold
            if cross_track_error > self.cross_track_threshold:
                use_fallback = True
                self.fallback_count += 1
        
        if use_fallback and self.mpc_solver is not None:
            # Use MPC as fallback
            p_vector = np.concatenate([aug_state, x_ref, y_ref, theta_ref])
            u_init = [0.01] * (self.collector.nu * self.collector.N)
            result = self.mpc_solver.run(p=p_vector, initial_guess=u_init)
            
            if result.exit_status == 'Converged':
                return result.solution[0:self.collector.nu]
            else:
                # If MPC fails, fall back to DNN
                pass
        
        # Use DNN
        with torch.no_grad():
            ref_tensor = torch.FloatTensor(ref_features).unsqueeze(0).to(self.device)
            control = self.model(ref_tensor)
            s_omega = control.cpu().numpy()[0]
        
        # Append v_ref as the third control input
        return np.array([s_omega[0], s_omega[1], self.v_ref])
    
    def get_fallback_stats(self):
        """Return statistics about MPC fallback usage."""
        if self.total_calls == 0:
            return {'fallback_rate': 0.0, 'fallback_count': 0, 'total_calls': 0}
        return {
            'fallback_rate': self.fallback_count / self.total_calls,
            'fallback_count': self.fallback_count,
            'total_calls': self.total_calls
        }
