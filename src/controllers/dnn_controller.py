import numpy as np
import torch
import torch.nn as nn


class MPCNeuralNetwork(nn.Module):
    """Deep Neural Network that mimics the MPC controller."""
    
    def __init__(self, state_dim=3, ref_feature_dim=42, hidden_dims=[512, 512, 256, 128], output_dim=3):
        super(MPCNeuralNetwork, self).__init__()

        # Add input normalization
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.register_buffer('ref_mean', torch.zeros(ref_feature_dim))
        self.register_buffer('ref_std', torch.ones(ref_feature_dim))
        
        input_dim = state_dim + ref_feature_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.15))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))  # Output: [s, omega, v]
        
        self.network = nn.Sequential(*layers)
        
        # Output constraints matching augmented MPC
        self.s_min, self.s_max = 0.02, 4.0
        self.w_min, self.w_max = -7.0, 7.0
        self.v_path_min, self.v_path_max = 0.01, 0.4

    def fit_normalization(self, states, refs):
        """Compute normalization statistics from training data"""
        self.state_mean = torch.FloatTensor(states.mean(axis=0))
        self.state_std = torch.FloatTensor(states.std(axis=0) + 1e-8)
        self.ref_mean = torch.FloatTensor(refs.mean(axis=0))
        self.ref_std = torch.FloatTensor(refs.std(axis=0) + 1e-8)

    def forward(self, state, ref_features):
        # Normalize inputs
        state = (state - self.state_mean) / self.state_std
        ref_features = (ref_features - self.ref_mean) / self.ref_std
        
        x = torch.cat([state, ref_features], dim=-1)
        output = self.network(x)
        
        # Scale outputs to match augmented MPC constraints
        s = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(output[:, 0:1])
        omega = self.w_max * torch.tanh(output[:, 1:2])
        v_path = self.v_path_min + (self.v_path_max - self.v_path_min) * torch.sigmoid(output[:, 2:3])
        
        return torch.cat([s, omega, v_path], dim=-1)


class DNNController:
    """Wrapper for using the trained DNN as a controller."""
    
    def __init__(self, model, collector, t_param):
        """
        Initialize DNN controller.
        
        Args:
            model: Trained neural network model
            collector: MPCDataCollector instance for reference generation
            t_param: Path parameter array
        """
        self.model = model
        self.collector = collector
        self.t_param = t_param
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
    
    def get_control(self, aug_state):
        """
        Get control action from DNN given current state.
        
        Args:
            aug_state: Augmented state [x, y, theta, theta_p]
        
        Returns:
            Control action [s, omega, v]
        """
        # The DNN was trained on geometric state, so we only pass the first 3 elements
        geometric_state = aug_state[:3]
        
        # Reference features are also based on the geometric state
        x_ref, y_ref, theta_ref = self.collector.get_reference_horizon(aug_state[3])
        ref_features = self.collector.extract_reference_features(geometric_state, x_ref, y_ref, theta_ref)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(geometric_state).unsqueeze(0).to(self.device)
            ref_tensor = torch.FloatTensor(ref_features).unsqueeze(0).to(self.device)
            control = self.model(state_tensor, ref_tensor)
            return control.cpu().numpy()[0]
