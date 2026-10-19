"""
Final comparison of MPC, LQR, and DNN controllers on the ellipse.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers.mpc_controller import MPCController
from controllers.lqr_controller import LQRController
from utils.reference import generate_ellipse_reference, find_projection_index


def final_comparison():
    """Compare MPC, LQR, and DNN on the ellipse."""
    print("="*70)
    print(" Final Controller Comparison: MPC vs LQR vs DNN")
    print("="*70)
    
    # Generate reference trajectory
    total_points = 10000
    t_param, x_ref, y_ref, theta_ref = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    
    # Calculate curvature
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    
    dt = 0.03
    v_ref = 0.2
    max_steps = 3000
    
    # Initial state
    initial_geometric_state = np.array([x_ref[0] + 0.1, y_ref[0] + 0.05, theta_ref[0] + 0.1])
    idx_closest = find_projection_index(initial_geometric_state, x_ref, y_ref)
    theta_p_initial = t_param[idx_closest]
    initial_aug_state = np.concatenate([initial_geometric_state, [theta_p_initial]])
    
    # =========================================
    # MPC Simulation
    # =========================================
    print(f"\n[1/3] Simulating MPC...")
    mpc = MPCController(nu=3, nx_aug=4, N=70, ts=dt, q=100, qtheta=10, 
                       r_control=0.1, r_v=0.001, v_ref=v_ref)
    mpc.build_optimizer(build_dir="my_optimizers")
    
    state = initial_aug_state.copy()
    traj_mpc = [state.copy()]
    ctrl_mpc = []
    err_mpc = []
    
    start_pos = state[:2]
    lap_mpc = False
    
    for step in range(max_steps):
        theta_p_current = state[3]
        N = 70
        theta_ref_horizon_p = theta_p_current + v_ref * np.arange(N) * dt
        
        x_ref_horizon = np.interp(theta_ref_horizon_p, t_param, x_ref)
        y_ref_horizon = np.interp(theta_ref_horizon_p, t_param, y_ref)
        theta_ref_unwrapped = np.unwrap(theta_ref)
        theta_ref_horizon = np.interp(theta_ref_horizon_p, t_param, theta_ref_unwrapped)
        theta_ref_horizon = (theta_ref_horizon + np.pi) % (2 * np.pi) - np.pi
        
        u = mpc.get_first_control(state, x_ref_horizon, y_ref_horizon, theta_ref_horizon)
        ctrl_mpc.append(u)
        
        idx = find_projection_index(state, x_ref, y_ref)
        error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
        err_mpc.append(error)
        
        s, omega, v_path = u[0], u[1], u[2]
        state = np.array([
            state[0] + dt * s * np.cos(state[2]),
            state[1] + dt * s * np.sin(state[2]),
            state[2] + dt * omega,
            state[3] + dt * v_path
        ])
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        state[3] = state[3] % (2 * np.pi)
        
        traj_mpc.append(state.copy())
        
        if step > 1000:
            dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
            if dist_to_start < 0.1:
                lap_mpc = True
                break
    
    traj_mpc = np.array(traj_mpc)
    ctrl_mpc = np.array(ctrl_mpc)
    err_mpc = np.array(err_mpc)
    print(f"      MPC: Mean error={np.mean(err_mpc)*1000:.2f}mm, Lap={lap_mpc}")
    
    # =========================================
    # LQR Simulation
    # =========================================
    print(f"\n[2/3] Simulating LQR...")
    lqr = LQRController(v_ref=v_ref, dt=dt)
    
    state = initial_geometric_state.copy()
    traj_lqr = [state.copy()]
    ctrl_lqr = []
    err_lqr = []
    
    start_pos = state[:2]
    lap_lqr = False
    
    for step in range(max_steps):
        idx = find_projection_index(state, x_ref, y_ref)
        path_state = np.array([x_ref[idx], y_ref[idx], theta_ref[idx], curvature[idx]])
        
        u = lqr.get_control(state, path_state)
        ctrl_lqr.append(u)
        
        error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
        err_lqr.append(error)
        
        v, omega = u[0], u[1]
        state = np.array([
            state[0] + dt * v * np.cos(state[2]),
            state[1] + dt * v * np.sin(state[2]),
            state[2] + dt * omega
        ])
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        
        traj_lqr.append(state.copy())
        
        if step > 1000:
            dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
            if dist_to_start < 0.1:
                lap_lqr = True
                break
    
    traj_lqr = np.array(traj_lqr)
    ctrl_lqr = np.array(ctrl_lqr)
    err_lqr = np.array(err_lqr)
    print(f"      LQR: Mean error={np.mean(err_lqr)*1000:.2f}mm, Lap={lap_lqr}")
    
    # =========================================
    # DNN Simulation (using pre-trained model)
    # =========================================
    print(f"\n[3/3] Simulating DNN...")
    import torch
    import torch.nn as nn
    
    class DNNController(nn.Module):
        def __init__(self, state_dim=3, ref_feature_dim=42, hidden_dims=[256, 256, 128, 64]):
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
                layers.append(nn.Dropout(0.15))
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
    
    # Load model
    model = DNNController()
    model.load_state_dict(torch.load('final_mpc_dnn_model.pth', weights_only=True))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load dataset for normalization
    data = np.load('mpc_dataset.npz')
    states = data['states']
    references = data['references']
    model.fit_normalization(states, references)
    
    class MockCollector:
        def __init__(self, t_param, x_ref, y_ref):
            self.t_param = t_param
            self.x_ref_array = x_ref
            self.y_ref_array = y_ref
    
    model.collector = MockCollector(t_param, x_ref, y_ref)
    model.N = 70
    model.ts = dt
    model.v_ref = v_ref
    
    # Simulate
    state = initial_geometric_state.copy()
    traj_dnn = [state.copy()]
    ctrl_dnn = []
    err_dnn = []
    
    start_pos = state[:2]
    lap_dnn = False
    
    with torch.no_grad():
        for step in range(max_steps):
            idx = find_projection_index(state, x_ref, y_ref)
            theta_p_current = t_param[idx]
            theta_ref_horizon_p = theta_p_current + v_ref * np.arange(70) * dt
            
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
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            ref_tensor = torch.FloatTensor(ref_features).unsqueeze(0).to(device)
            u = model(state_tensor, ref_tensor).cpu().numpy()[0]
            ctrl_dnn.append(u)
            
            error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
            err_dnn.append(error)
            
            s, omega, v_path = u[0], u[1], u[2]
            state = np.array([
                state[0] + dt * s * np.cos(state[2]),
                state[1] + dt * s * np.sin(state[2]),
                state[2] + dt * omega
            ])
            state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
            
            traj_dnn.append(state.copy())
            
            if step > 1000:
                dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
                if dist_to_start < 0.1:
                    lap_dnn = True
                    break
    
    traj_dnn = np.array(traj_dnn)
    ctrl_dnn = np.array(ctrl_dnn)
    err_dnn = np.array(err_dnn)
    print(f"      DNN: Mean error={np.mean(err_dnn)*1000:.2f}mm, Lap={lap_dnn}")
    
    # =========================================
    # Visualization
    # =========================================
    print(f"\n[4/4] Generating visualization...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: All trajectories
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Reference')
    ax1.plot(traj_mpc[:, 0], traj_mpc[:, 1], 'b-', linewidth=2, label='MPC', alpha=0.8)
    ax1.plot(traj_lqr[:, 0], traj_lqr[:, 1], 'g-', linewidth=2, label='LQR', alpha=0.8)
    ax1.plot(traj_dnn[:, 0], traj_dnn[:, 1], 'r-', linewidth=2, label='DNN', alpha=0.8)
    ax1.plot(initial_aug_state[0], initial_aug_state[1], 'ko', markersize=10, label='Start')
    ax1.set_xlabel('x [m]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y [m]', fontsize=11, fontweight='bold')
    ax1.set_title('All Trajectories on Ellipse', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Tracking errors
    ax2 = plt.subplot(2, 3, 2)
    times_mpc = np.arange(len(err_mpc)) * dt
    times_lqr = np.arange(len(err_lqr)) * dt
    times_dnn = np.arange(len(err_dnn)) * dt
    ax2.plot(times_mpc, err_mpc * 1000, 'b-', linewidth=2, label='MPC')
    ax2.plot(times_lqr, err_lqr * 1000, 'g-', linewidth=2, label='LQR')
    ax2.plot(times_dnn, err_dnn * 1000, 'r-', linewidth=2, label='DNN')
    ax2.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax2.set_title('Tracking Error Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Linear velocity
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times_mpc, ctrl_mpc[:, 0], 'b-', linewidth=2, label='MPC')
    ax3.plot(times_lqr, ctrl_lqr[:, 0], 'g-', linewidth=2, label='LQR')
    ax3.plot(times_dnn, ctrl_dnn[:, 0], 'r-', linewidth=2, label='DNN')
    ax3.axhline(y=v_ref, color='k', linestyle='--', alpha=0.5, label='Reference')
    ax3.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Linear Velocity [m/s]', fontsize=11, fontweight='bold')
    ax3.set_title('Linear Velocity Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Angular velocity
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(times_mpc, ctrl_mpc[:, 1], 'b-', linewidth=2, label='MPC')
    ax4.plot(times_lqr, ctrl_lqr[:, 1], 'g-', linewidth=2, label='LQR')
    ax4.plot(times_dnn, ctrl_dnn[:, 1], 'r-', linewidth=2, label='DNN')
    ax4.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Angular Velocity [rad/s]', fontsize=11, fontweight='bold')
    ax4.set_title('Angular Velocity Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Error distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(err_mpc * 1000, bins=30, alpha=0.5, label='MPC', color='blue')
    ax5.hist(err_lqr * 1000, bins=30, alpha=0.5, label='LQR', color='green')
    ax5.hist(err_dnn * 1000, bins=30, alpha=0.5, label='DNN', color='red')
    ax5.set_xlabel('Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance bar chart
    ax6 = plt.subplot(2, 3, 6)
    controllers = ['MPC', 'LQR', 'DNN']
    mean_errors = [np.mean(err_mpc)*1000, np.mean(err_lqr)*1000, np.mean(err_dnn)*1000]
    lap_status = [lap_mpc, lap_lqr, lap_dnn]
    colors = ['blue', 'green', 'red']
    bars = ax6.bar(controllers, mean_errors, color=colors, alpha=0.7)
    ax6.set_ylabel('Mean Tracking Error [mm]', fontsize=11, fontweight='bold')
    ax6.set_title('Mean Error Comparison', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add lap status labels
    for i, (bar, lap) in enumerate(zip(bars, lap_status)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{"✓" if lap else "✗"}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.suptitle('Final Controller Comparison: MPC vs LQR vs DNN', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/final_controller_comparison.png', dpi=200, bbox_inches='tight')
    
    print(f"      Visualization saved to 'outputs/final_controller_comparison.png'")
    
    # Summary table
    print("\n" + "="*70)
    print(" Final Summary")
    print("="*70)
    print(f"{'Controller':<12} {'Mean Error':<15} {'Max Error':<15} {'Lap':<10}")
    print("-" * 70)
    print(f"{'MPC':<12} {np.mean(err_mpc)*1000:<15.2f} {np.max(err_mpc)*1000:<15.2f} {str(lap_mpc):<10}")
    print(f"{'LQR':<12} {np.mean(err_lqr)*1000:<15.2f} {np.max(err_lqr)*1000:<15.2f} {str(lap_lqr):<10}")
    print(f"{'DNN':<12} {np.mean(err_dnn)*1000:<15.2f} {np.max(err_dnn)*1000:<15.2f} {str(lap_dnn):<10}")
    print("="*70)
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    final_comparison()
