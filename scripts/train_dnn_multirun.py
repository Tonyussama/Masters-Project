"""
Multi-run DNN training with fixed seeds for robust statistics.
"""

import opengen as og
import casadi.casadi as cs
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.reference import generate_ellipse_reference, find_projection_index
from utils.data_collection import MPCDataCollector


class MPCDataset(Dataset):
    def __init__(self, states, references, controls):
        self.states = torch.FloatTensor(states)
        self.references = torch.FloatTensor(references)
        self.controls = torch.FloatTensor(controls)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.references[idx], self.controls[idx]


class MPCNeuralNetwork(nn.Module):
    def __init__(self, state_dim=3, ref_feature_dim=42, hidden_dims=[1024, 1024, 512, 256, 128], output_dim=3):
        super(MPCNeuralNetwork, self).__init__()
        
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
            # No dropout - removed for better learning
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
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


def train_model(model, train_loader, val_loader, states, references, epochs=1000, lr=0.001, seed=42):
    """Train the neural network with fixed seed."""
    
    # Set seed for this run
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model.fit_normalization(states, references)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Training on {device}")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     patience=15, factor=0.5, verbose=False)
    criterion = nn.SmoothL1Loss()  # Huber loss
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 40
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for states, refs, controls in train_loader:
            states, refs, controls = states.to(device), refs.to(device), controls.to(device)
            
            optimizer.zero_grad()
            predictions = model(states, refs)
            loss = criterion(predictions, controls)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, refs, controls in val_loader:
                states, refs, controls = states.to(device), refs.to(device), controls.to(device)
                predictions = model(states, refs)
                loss = criterion(predictions, controls)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_mpc_dnn_model_seed{seed}.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"      Epoch [{epoch+1}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
        
        if patience_counter >= early_stop_patience:
            print(f"      Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(f'best_mpc_dnn_model_seed{seed}.pth'))
    print(f"      Training complete! Best validation loss: {best_val_loss:.4f}")
    
    return model, best_val_loss


def test_dnn_on_ellipse(model, t_param, x_ref, y_ref, theta_ref, seed):
    """Test DNN on ellipse and return RMS error."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
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
    model.ts = 0.03
    model.v_ref = 0.2
    
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
            idx = find_projection_index(state, x_ref, y_ref)
            theta_p_current = t_param[idx]
            theta_ref_horizon_p = theta_p_current + model.v_ref * np.arange(70) * dt
            
            x_ref_horizon = np.interp(theta_ref_horizon_p, t_param, x_ref)
            y_ref_horizon = np.interp(theta_ref_horizon_p, t_param, y_ref)
            theta_ref_unwrapped = np.unwrap(theta_ref)
            theta_ref_horizon = np.interp(theta_ref_horizon_p, t_param, theta_ref_unwrapped)
            theta_ref_horizon = (theta_ref_horizon + np.pi) % (2 * np.pi) - np.pi
            
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
            controls.append(u)
            
            error = np.sqrt((state[0] - x_ref[idx])**2 + (state[1] - y_ref[idx])**2)
            errors.append(error)
            
            # Use only first two outputs (s and omega)
            s, omega = u[0], u[1]
            state = np.array([
                state[0] + dt * s * np.cos(state[2]),
                state[1] + dt * s * np.sin(state[2]),
                state[2] + dt * omega
            ])
            state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
            
            trajectory.append(state.copy())
            
            if step > min_steps_for_lap:
                dist_to_start = np.sqrt((state[0] - start_pos[0])**2 + (state[1] - start_pos[1])**2)
                if dist_to_start < lap_completion_threshold:
                    completed_lap = True
                    break
    
    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    lap_time = len(controls) * dt
    
    return {
        'rms_error_mm': rms_error * 1000,
        'mean_error_mm': mean_error * 1000,
        'max_error_mm': max_error * 1000,
        'lap_completed': completed_lap,
        'lap_time_s': lap_time,
        'steps': len(controls)
    }


def multirun_training(num_runs=5):
    """Run training multiple times with different seeds."""
    print("="*70)
    print(f" Multi-Run DNN Training ({num_runs} runs)")
    print("="*70)
    
    # Generate reference trajectory
    total_points = 10000
    t_param, x_ref, y_ref, theta_ref = generate_ellipse_reference(
        total_points=total_points, height=4.0, width=0.2
    )
    
    # Build MPC optimizer
    nu = 3
    nx_aug = 4
    N = 70
    ts = 0.03
    q = 100
    qtheta = 10
    r_control = 0.1
    r_v = 0.001
    v_ref = 0.2
    
    print(f"\n[1/4] Building MPC optimizer (release mode)...")
    from controllers.mpc_controller import MPCController
    mpc = MPCController(nu=nu, nx_aug=nx_aug, N=N, ts=ts,
                       q=q, qtheta=qtheta, r_control=r_control, r_v=r_v, v_ref=v_ref)
    mpc.build_optimizer(build_dir="my_optimizers")
    print("      ✓ Release build complete")
    
    # Add optimizer to path
    sys.path.insert(1, "my_optimizers/path_following_aug")
    
    # Collect data (once, reuse for all runs)
    print(f"\n[2/4] Collecting training data (5000 anchors)...")
    collector = MPCDataCollector(mpc.solver, t_param, x_ref, y_ref, theta_ref, N, nu, ts, v_ref)
    states, references, controls = collector.collect_data_corridor(num_anchors=5000, offset_distance=0.1)
    print(f"      ✓ Collected {len(states)} samples")
    
    # Prepare training data
    print(f"\n[3/4] Preparing training data...")
    dataset = MPCDataset(states, references, controls)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    ref_feature_dim = references.shape[1]
    print(f"      Training: {train_size} samples | Validation: {val_size} samples")
    
    # Run multiple training runs
    print(f"\n[4/4] Running {num_runs} training runs with different seeds...")
    results = []
    seeds = [42, 123, 456, 789, 101112]  # Different seeds for each run
    
    for i, seed in enumerate(seeds[:num_runs]):
        print(f"\n--- Run {i+1}/{num_runs} (seed={seed}) ---")
        
        model = MPCNeuralNetwork(state_dim=3, ref_feature_dim=ref_feature_dim, 
                                hidden_dims=[1024, 1024, 512, 256, 128], output_dim=3)
        
        trained_model, best_val_loss = train_model(model, train_loader, val_loader, 
                                                   states, references, epochs=1000, lr=0.001, seed=seed)
        
        # Test on ellipse
        test_results = test_dnn_on_ellipse(trained_model, t_param, x_ref, y_ref, theta_ref, seed)
        test_results['seed'] = seed
        test_results['val_loss'] = best_val_loss
        results.append(test_results)
        
        print(f"      RMS Error: {test_results['rms_error_mm']:.2f} mm")
        print(f"      Max Error: {test_results['max_error_mm']:.2f} mm")
        print(f"      Lap Completed: {test_results['lap_completed']}")
        print(f"      Lap Time: {test_results['lap_time_s']:.2f} s")
    
    # Summary statistics
    print("\n" + "="*70)
    print(" Summary Statistics")
    print("="*70)
    
    rms_errors = [r['rms_error_mm'] for r in results]
    max_errors = [r['max_error_mm'] for r in results]
    lap_times = [r['lap_time_s'] for r in results]
    
    print(f"\nRMS Error:")
    print(f"  Mean: {np.mean(rms_errors):.2f} mm")
    print(f"  Std: {np.std(rms_errors):.2f} mm")
    print(f"  Min: {np.min(rms_errors):.2f} mm")
    print(f"  Max: {np.max(rms_errors):.2f} mm")
    
    print(f"\nMax Error:")
    print(f"  Mean: {np.mean(max_errors):.2f} mm")
    print(f"  Std: {np.std(max_errors):.2f} mm")
    print(f"  Min: {np.min(max_errors):.2f} mm")
    print(f"  Max: {np.max(max_errors):.2f} mm")
    
    print(f"\nLap Time:")
    print(f"  Mean: {np.mean(lap_times):.2f} s")
    print(f"  Std: {np.std(lap_times):.2f} s")
    
    print(f"\nLap Completion:")
    completed_count = sum(1 for r in results if r['lap_completed'])
    print(f"  {completed_count}/{num_runs} runs completed lap")
    
    print("\n" + "="*70)
    print(" Multi-Run Training Complete!")
    print("="*70)
    
    # Save best model
    best_idx = np.argmin(rms_errors)
    best_seed = results[best_idx]['seed']
    print(f"\nBest model: seed={best_seed}, RMS={results[best_idx]['rms_error_mm']:.2f} mm")
    print(f"Saved as: best_mpc_dnn_model_seed{best_seed}.pth")
    
    return results


if __name__ == "__main__":
    num_runs = 3  # Start with 3 runs for faster results
    results = multirun_training(num_runs=num_runs)
