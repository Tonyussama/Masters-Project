"""
Test and visualize DNN controller performance on the ellipse.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.reference import generate_ellipse_reference, find_projection_index
from utils.data_collection import MPCDataCollector


# Load the trained model
class DNNController(nn.Module):
    def __init__(self, state_dim=3, ref_feature_dim=42, hidden_dims=[1024, 1024, 512, 256, 128]):
        super().__init__()
        self.state_dim = state_dim
        self.ref_feature_dim = ref_feature_dim
        
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
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
        
        self.s_min, self.s_max = 0.02, 4.0
        self.w_min, self.w_max = -7.0, 7.0
        self.v_path_min, self.v_path_max = 0.01, 0.4

    def fit_normalization(self, states, refs):
        self.state_mean = torch.FloatTensor(states.mean(axis=0))
        self.state_std = torch.FloatTensor(states.std(axis=0) + 1e-8)
        self.ref_mean = torch.FloatTensor(refs.mean(axis=0))
        self.ref_std = torch.FloatTensor(refs.std(axis=0) + 1e-8)

    def forward(self, state, ref_features):
        state = (state - self.state_mean) / self.state_std
        ref_features = (ref_features - self.ref_mean) / self.ref_std
        
        x = torch.cat([state, ref_features], dim=-1)
        output = self.network(x)
        
        s = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(output[:, 0:1])
        omega = self.w_max * torch.tanh(output[:, 1:2])
        v_path = self.v_path_min + (self.v_path_max - self.v_path_min) * torch.sigmoid(output[:, 2:3])
        
        return torch.cat([s, omega, v_path], dim=-1)


def simulate_dnn_on_ellipse():
    """Simulate DNN controller on complete ellipse."""
    print("="*70)
    print(" DNN Controller on Ellipse")
    print("="*70)
    
    # Load reference trajectory
    total_points = 10000
    t_param, x_ref, y_ref, theta_ref = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    
    # Load trained model
    print("\n[1/4] Loading trained DNN model...")
    model = DNNController()
    model.load_state_dict(torch.load('final_mpc_dnn_model.pth', weights_only=True))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("      Model loaded successfully")
    
    # Load dataset for normalization
    print("\n[2/4] Loading dataset for normalization...")
    data = np.load('mpc_dataset.npz')
    states = data['states']
    references = data['references']
    model.fit_normalization(states, references)
    print("      Normalization fitted")
    
    # Create mock collector for reference data
    class MockCollector:
        def __init__(self, t_param, x_ref, y_ref):
            self.t_param = t_param
            self.x_ref_array = x_ref
            self.y_ref_array = y_ref
    
    model.collector = MockCollector(t_param, x_ref, y_ref)
    model.N = 70
    model.ts = 0.03
    model.v_ref = 0.2
    
    # Simulate DNN on ellipse
    print("\n[3/4] Simulating DNN on ellipse...")
    
    # Initial state
    initial_state = np.array([x_ref[0] + 0.1, y_ref[0] + 0.05, theta_ref[0] + 0.1])
    
    state = initial_state.copy()
    trajectory = [state.copy()]
    controls = []
    errors = []
    
    dt = 0.03
    max_steps = 3000
    min_steps_for_lap = 1000
    lap_completion_threshold = 0.1
    start_pos = state[:2]
    completed_lap = False
    
    with torch.no_grad():
        for step in range(max_steps):
            # Get reference horizon
            idx = find_projection_index(state, x_ref, y_ref)
            theta_p_current = t_param[idx]
            N = 70
            v_ref = 0.2
            theta_ref_horizon_p = theta_p_current + v_ref * np.arange(N) * dt
            
            x_ref_horizon = np.interp(theta_ref_horizon_p, t_param, x_ref)
            y_ref_horizon = np.interp(theta_ref_horizon_p, t_param, y_ref)
            theta_ref_unwrapped = np.unwrap(theta_ref)
            theta_ref_horizon = np.interp(theta_ref_horizon_p, t_param, theta_ref_unwrapped)
            theta_ref_horizon = (theta_ref_horizon + np.pi) % (2 * np.pi) - np.pi
            
            # Extract reference features
            x, y, theta_state = state
            cos_theta = np.cos(-theta_state)
            sin_theta = np.sin(-theta_state)
            
            dx = x_ref_horizon - x
            dy = y_ref_horizon - y
            
            x_ref_local = dx * cos_theta - dy * sin_theta
            y_ref_local = dx * sin_theta + dy * cos_theta
            theta_ref_local = theta_ref_horizon - theta_state
            theta_ref_local = np.arctan2(np.sin(theta_ref_local), np.cos(theta_ref_local))
            
            stride = 5
            ref_features = np.concatenate([
                x_ref_local[::stride],
                y_ref_local[::stride],
                theta_ref_local[::stride]
            ])
            
            # Get control from DNN
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            ref_tensor = torch.FloatTensor(ref_features).unsqueeze(0).to(device)
            u = model(state_tensor, ref_tensor).cpu().numpy()[0]
            controls.append(u)
            
            # Calculate error
            error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
            errors.append(error)
            
            # Apply dynamics - use only first two outputs (s and omega)
            s, omega = u[0], u[1]
            state = np.array([
                state[0] + dt * s * np.cos(state[2]),
                state[1] + dt * s * np.sin(state[2]),
                state[2] + dt * omega
            ])
            state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
            
            trajectory.append(state.copy())
            
            # Check lap completion
            if step > min_steps_for_lap:
                dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
                if dist_to_start < lap_completion_threshold:
                    completed_lap = True
                    print(f"      Lap completed at step {step + 1}")
                    break
    
    trajectory = np.array(trajectory)
    controls = np.array(controls)
    errors = np.array(errors)
    times = np.arange(len(controls)) * dt
    
    print(f"      Simulation complete: {len(controls)} steps")
    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    mean_error = np.mean(errors)
    print(f"      RMS tracking error: {rms_error*1000:.2f} mm")
    print(f"      Mean tracking error: {mean_error*1000:.2f} mm")
    print(f"      Max tracking error: {np.max(errors)*1000:.2f} mm")
    print(f"      Lap completed: {completed_lap}")
    print(f"      Lap time: {len(controls)*dt:.2f} s")
    
    # Visualization
    print("\n[4/4] Generating visualization...")
    
    fig = plt.figure(figsize=(20, 10))
    
    # Plot 1: Trajectory with error vectors
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.6, label='Reference', zorder=1)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='DNN', alpha=0.8, zorder=2)
    
    # Draw error vectors every 50 steps
    error_step = 50
    for i in range(0, len(trajectory), error_step):
        idx = find_projection_index(trajectory[i], x_ref, y_ref)
        ax1.plot([trajectory[i, 0], x_ref[idx]], [trajectory[i, 1], y_ref[idx]], 
                'b-', linewidth=1, alpha=0.3, zorder=0)
    
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='Start', zorder=3)
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'bs', markersize=12, label='End', zorder=3)
    ax1.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax1.set_title('DNN Trajectory with Error Vectors', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 1b: Zoom in on top section (high curvature)
    ax1b = plt.subplot(2, 4, 2)
    ax1b.plot(x_ref, y_ref, 'k--', linewidth=3, alpha=0.6, label='Reference', zorder=1)
    ax1b.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=3, label='DNN', alpha=0.9, zorder=2)
    ax1b.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax1b.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax1b.set_title('Zoom: Top Section (High Curvature)', fontsize=12, fontweight='bold')
    ax1b.legend(fontsize=9)
    ax1b.grid(True, alpha=0.3)
    ax1b.axis('equal')
    ax1b.set_xlim([-0.15, 0.15])
    ax1b.set_ylim([1.8, 2.2])
    
    # Plot 2: Tracking error
    ax2 = plt.subplot(2, 4, 3)
    ax2.plot(times, errors * 1000, 'b-', linewidth=2)
    ax2.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax2.set_title('Tracking Error vs Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Forward speed
    ax3 = plt.subplot(2, 4, 4)
    ax3.plot(times, controls[:, 0], 'b-', linewidth=2)
    ax3.axhline(y=0.2, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax3.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Forward Speed [m/s]', fontsize=11, fontweight='bold')
    ax3.set_title('Forward Speed vs Time', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Angular velocity
    ax4 = plt.subplot(2, 4, 5)
    ax4.plot(times, controls[:, 1], 'r-', linewidth=2)
    ax4.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Angular Velocity [rad/s]', fontsize=11, fontweight='bold')
    ax4.set_title('Angular Velocity vs Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Path velocity
    ax5 = plt.subplot(2, 4, 6)
    ax5.plot(times, controls[:, 2], 'g-', linewidth=2)
    ax5.axhline(y=0.2, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax5.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Path Velocity [rad/s]', fontsize=11, fontweight='bold')
    ax5.set_title('Path Velocity vs Time', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error distribution
    ax6 = plt.subplot(2, 4, 7)
    ax6.hist(errors * 1000, bins=30, alpha=0.7, color='red')
    ax6.set_xlabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Error vs path parameter
    ax7 = plt.subplot(2, 4, 8)
    theta_p_trajectory = []
    for i in range(len(errors)):  # Match length with errors
        idx = find_projection_index(trajectory[i], x_ref, y_ref)
        theta_p_trajectory.append(t_param[idx])
    ax7.plot(theta_p_trajectory, errors * 1000, 'r-', linewidth=2)
    ax7.set_xlabel('Path Parameter [rad]', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax7.set_title('Error vs Path Parameter', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('DNN Controller Performance on Ellipse', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/dnn_ellipse_performance.png', dpi=200, bbox_inches='tight')
    
    print(f"      Visualization saved to 'outputs/dnn_ellipse_performance.png'")
    
    # Summary
    print("\n" + "="*70)
    print(" Summary")
    print("="*70)
    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    mean_error = np.mean(errors)
    print(f"RMS tracking error: {rms_error*1000:.2f} mm")
    print(f"Mean tracking error: {mean_error*1000:.2f} mm")
    print(f"Max tracking error: {np.max(errors)*1000:.2f} mm")
    print(f"Lap completed: {completed_lap}")
    print(f"Lap time: {len(controls)*dt:.2f} s")
    print(f"Steps: {len(controls)}")
    print("="*70)


if __name__ == "__main__":
    simulate_dnn_on_ellipse()
