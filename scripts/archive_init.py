import opengen as og
import casadi.casadi as cs
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# Set fixed seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.reference import generate_ellipse_reference, find_projection_index
from utils.data_collection import MPCDataCollector

# =========================================
# PART 1: MPC SETUP
# =========================================

print("="*70)
print(" MPC to DNN: Complete Training Pipeline")
print("="*70)

# Define basic parameters
(nu, nx_aug) = (3, 4)  # nu=3 inputs [s, ω, v], nx_aug=4 states [x, y, θ, θₚ]
N = 70
ts = 0.03

# Cost weights from Final_Shared.py
q = 100
qtheta = 10
r_control = 0.1
r_v = 0.001
v_ref = 0.2

# Create reference trajectory (ellipse)
total_points = 10000
t_param, x_ref_array, y_ref_array, theta_ref_array = generate_ellipse_reference(
    total_points=total_points, height=4.0, width=0.2
)

print(f"\n[1/7] Reference trajectory created: {len(x_ref_array)} points")

# Define MPC optimization problem
u = cs.SX.sym('u', nu * N)
p = cs.SX.sym('p', nx_aug + 3 * N)

x0 = p[0]
y0 = p[1]
theta0 = p[2]
theta_p0 = p[3]

x_ref_sym = p[nx_aug : nx_aug + N]
y_ref_sym = p[nx_aug + N : nx_aug + 2 * N]
theta_ref_sym = p[nx_aug + 2 * N : nx_aug + 3 * N]

# Build cost function
cost = 0
x = x0
y = y0
theta = theta0
theta_p = theta_p0

for t_step in range(N):
    xref_t = x_ref_sym[t_step]
    yref_t = y_ref_sym[t_step]
    thetaref_t = theta_ref_sym[t_step]

    cost += q * ((x - xref_t)**2 + (y - yref_t)**2) 
    heading_error = cs.atan2(cs.sin(theta - thetaref_t), cs.cos(theta - thetaref_t))
    cost += qtheta * (heading_error)**2

    s_k = u[t_step * nu]
    w = u[t_step * nu + 1]
    v_k = u[t_step * nu + 2]
    cost += r_control * (s_k**2 + w**2) + r_v * (v_k - v_ref)**2

    x = x + ts * s_k * cs.cos(theta)
    y = y + ts * s_k * cs.sin(theta)
    theta = theta + ts * w
    theta_p = theta_p + ts * v_k

# Terminal cost
xref_N = x_ref_sym[N - 1]
yref_N = y_ref_sym[N - 1]
thetaref_N = theta_ref_sym[N - 1]
terminal_pos_error = (x - xref_N)**2 + (y - yref_N)**2
terminal_heading_error = cs.atan2(cs.sin(theta - thetaref_N), cs.cos(theta - thetaref_N))
cost += q * terminal_pos_error + qtheta * terminal_heading_error**2

# Constraints
umin = []
umax = []
for _ in range(N):
    umin.extend([0.02, -7.0, 0.01])  # Bounds for [s, ω, v]
    umax.extend([4.0, 7.0, 0.4])     # Bounds for [s, ω, v]

bounds = og.constraints.Rectangle(umin, umax)

# Build optimizer
print("[2/7] Building MPC optimizer...")
problem = og.builder.Problem(u, p, cost).with_constraints(bounds)

build_config = og.config.BuildConfiguration() \
    .with_build_directory("my_optimizers") \
    .with_build_mode("debug") \
    .with_tcp_interface_config() \
    .with_build_python_bindings()

meta = og.config.OptimizerMeta().with_optimizer_name("path_following_aug")
solver_config = og.config.SolverConfiguration().with_tolerance(1e-5)

builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config)
builder.build()

# Import solver
sys.path.insert(0, "my_optimizers/path_following_aug")
import path_following_aug
solver = path_following_aug.solver()

print("      MPC optimizer ready!")

# =========================================
# PART 2: DATA COLLECTION FROM MPC
# =========================================

# Collect data using corridor experiment methodology (large dataset)
collector = MPCDataCollector(solver, t_param, x_ref_array, y_ref_array, theta_ref_array, N, nu, ts, v_ref)
states, references, controls = collector.collect_data_corridor(num_anchors=5000, offset_distance=0.1)

# Save dataset
np.savez('mpc_dataset.npz', 
         states=states, 
         references=references, 
         controls=controls,
         x_ref_array=x_ref_array,
         y_ref_array=y_ref_array,
         t_param=t_param,
         theta_ref_array=theta_ref_array,
         N=N, nu=nu, ts=ts)
print("      Dataset saved to 'mpc_dataset.npz'")

# =========================================
# PART 3: NEURAL NETWORK DEFINITION
# =========================================

class MPCDataset(Dataset):
    """PyTorch dataset for MPC data."""
    def __init__(self, states, references, controls):
        self.states = torch.FloatTensor(states)
        self.references = torch.FloatTensor(references)
        self.controls = torch.FloatTensor(controls)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.references[idx], self.controls[idx]


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
            # layers.append(nn.Dropout(0.15))  # Removed dropout for better learning
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


# =========================================
# PART 4: TRAINING FUNCTION
# =========================================

def train_model(model, train_loader, val_loader, states, references, epochs=300, lr=0.001):
    """Train the neural network to mimic MPC."""




    model.fit_normalization(states, references)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[4/7] Training neural network on {device}")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     patience=15, factor=0.5, verbose=False)
    criterion = nn.SmoothL1Loss()  # Huber loss - less sensitive to outliers
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 40
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for states, refs, controls in train_loader:
            states, refs, controls = states.to(device), refs.to(device), controls.to(device)
            
            optimizer.zero_grad()
            predictions = model(states, refs)
            loss = criterion(predictions, controls)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
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
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_mpc_dnn_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"      Epoch [{epoch+1}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
        
        if patience_counter >= early_stop_patience:
            print(f"      Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_mpc_dnn_model.pth'))
    print(f"      Training complete! Best validation loss: {best_val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2.5, alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', linewidth=2.5, alpha=0.8)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('MSE Loss', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', linewidth=2.5, alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', linewidth=2.5, alpha=0.8)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('MSE Loss (log scale)', fontsize=13)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title('Training Progress (Log Scale)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    return model


# =========================================
# PART 5: DNN CONTROLLER
# =========================================

class DNNController:
    """Wrapper for using the trained DNN as a controller."""
    def __init__(self, model, collector, t_param):
        self.model = model
        self.collector = collector
        self.t_param = t_param
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
    
    def get_control(self, aug_state):
        """Get control action from DNN given current state."""
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


def simulate_trajectory(controller, initial_state, ts, steps=300):
    """Simulate closed-loop trajectory."""
    trajectory = [initial_state.copy().astype(np.float64)]
    controls = []
    
    # Convert initial geometric state to augmented state
    idx_closest = find_projection_index(initial_state, controller.collector.x_ref_array, controller.collector.y_ref_array)
    theta_p_initial = controller.t_param[idx_closest]
    aug_current = np.concatenate([initial_state.astype(np.float64), [theta_p_initial]]).astype(np.float64)
    
    for _ in range(steps):
        u = controller.get_control(aug_current)
        u = np.array(u).flatten()
        controls.append(u.tolist())
        
        s, w, v_path = u[0], u[1], u[2]
        
        # Apply dynamics
        aug_current = np.array([
            aug_current[0] + ts * s * np.cos(aug_current[2]),
            aug_current[1] + ts * s * np.sin(aug_current[2]),
            aug_current[2] + ts * w,
            aug_current[3] + ts * v_path
        ], dtype=np.float64)
        # Wrap angle and path parameter
        aug_current[2] = (aug_current[2] + np.pi) % (2 * np.pi) - np.pi
        aug_current[3] = aug_current[3] % (2 * np.pi)
        trajectory.append(aug_current.copy().astype(np.float64))
    
    return np.array(trajectory), np.array(controls)


# =========================================
# PART 6: VISUALIZATION
# =========================================

def plot_comprehensive_results(trajectories, controls_list, x_ref_array, y_ref_array, initial_states):
    """Create comprehensive visualization of results."""
    
    fig = plt.figure(figsize=(18, 11))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    # Plot 1: Full trajectories
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(x_ref_array, y_ref_array, 'k--', linewidth=3, alpha=0.5, label='Reference Path')
    for i, (traj, init_state) in enumerate(zip(trajectories, initial_states)):
        plt.plot(traj[:, 0], traj[:, 1], '-', color=colors[i], 
                label=f'Init {i+1}', linewidth=2.5, alpha=0.85)
        plt.plot(init_state[0], init_state[1], 'o', color=colors[i],
                markersize=12, markeredgecolor='black', markeredgewidth=2)
    plt.xlabel('x [m]', fontsize=12, fontweight='bold')
    plt.ylabel('y [m]', fontsize=12, fontweight='bold')
    plt.title('Full Trajectories', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot 2: Zoomed initial phase
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(x_ref_array, y_ref_array, 'k--', linewidth=3, alpha=0.5)
    for i, (traj, init_state) in enumerate(zip(trajectories, initial_states)):
        plt.plot(traj[:60, 0], traj[:60, 1], '-', color=colors[i], linewidth=2.5, alpha=0.85)
        plt.plot(init_state[0], init_state[1], 'o', color=colors[i],
                markersize=12, markeredgecolor='black', markeredgewidth=2)
    plt.xlabel('x [m]', fontsize=12, fontweight='bold')
    plt.ylabel('y [m]', fontsize=12, fontweight='bold')
    plt.title('Initial Tracking (First 60 steps)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot 3: Tracking errors over time
    ax3 = plt.subplot(2, 3, 3)
    for i, traj in enumerate(trajectories):
        errors = []
        for state in traj:
            idx = find_projection_index(state, x_ref_array, y_ref_array)
            error = np.sqrt((state[0] - x_ref_array[idx])**2 + 
                          (state[1] - y_ref_array[idx])**2)
            errors.append(error)
        time = np.arange(len(errors)) * ts
        plt.plot(time, errors, '-', color=colors[i], linewidth=2, alpha=0.8)
    plt.xlabel('Time [s]', fontsize=12, fontweight='bold')
    plt.ylabel('Tracking Error [m]', fontsize=12, fontweight='bold')
    plt.title('Tracking Error Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Linear velocity
    ax4 = plt.subplot(2, 3, 4)
    for i, controls in enumerate(controls_list):
        time = np.arange(len(controls)) * ts
        plt.plot(time, controls[:, 0], '-', color=colors[i], linewidth=2, alpha=0.8)
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Forward Speed (s) [m/s]', fontsize=12)
    plt.title('Control: Forward Speed', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 4.1])
    
    # Plot 5: Angular velocity
    ax5 = plt.subplot(2, 3, 5)
    for i, controls in enumerate(controls_list):
        time = np.arange(len(controls)) * ts
        plt.plot(time, controls[:, 1], '-', color=colors[i], linewidth=2, alpha=0.8)
    plt.xlabel('Time [s]', fontsize=12, fontweight='bold')
    plt.ylabel('Angular Velocity (ω) [rad/s]', fontsize=12, fontweight='bold')
    plt.title('Control: Angular Velocity', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Path parameter velocity
    ax6 = plt.subplot(2, 3, 6)
    for i, controls in enumerate(controls_list):
        time = np.arange(len(controls)) * ts
        plt.plot(time, controls[:, 2], '-', color=colors[i], linewidth=2, alpha=0.8)
    plt.xlabel('Time [s]', fontsize=12, fontweight='bold')
    plt.ylabel('Path Velocity (v) [rad/s]', fontsize=12, fontweight='bold')
    plt.title('Control: Path Parameter Velocity', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 0.5])
    
    plt.suptitle('DNN Controller Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('dnn_results_comprehensive.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\n" + "="*70)
    print(" Tracking Performance Statistics")
    print("="*70)
    for i, traj in enumerate(trajectories):
        errors = []
        for state in traj:
            idx = find_projection_index(state, x_ref_array, y_ref_array)
            error = np.sqrt((state[0] - x_ref_array[idx])**2 + 
                          (state[1] - y_ref_array[idx])**2)
            errors.append(error)
        
        errors = np.array(errors)
        print(f"\nTrajectory {i+1} (Initial: [{initial_states[i][0]:.2f}, {initial_states[i][1]:.2f}, {initial_states[i][2]:.2f}]):")
        print(f"  Mean error:   {np.mean(errors)*1000:.2f} mm")
        print(f"  Max error:    {np.max(errors)*1000:.2f} mm")
        print(f"  Std error:    {np.std(errors)*1000:.2f} mm")
        print(f"  Final error:  {errors[-1]*1000:.2f} mm")


# =========================================
# MAIN EXECUTION
# =========================================

print("\n[5/7] Preparing training data...")
dataset = MPCDataset(states, references, controls)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, drop_last=True)

print(f"      Training: {train_size} samples | Validation: {val_size} samples")

# Create and train model
ref_feature_dim = references.shape[1]
print(f"      Network input: state(3) + references({ref_feature_dim}) = {3 + ref_feature_dim} dims")

model = MPCNeuralNetwork(state_dim=3, ref_feature_dim=ref_feature_dim, 
                        hidden_dims=[1024, 1024, 512, 256, 128], output_dim=3)

trained_model = train_model(model, train_loader, val_loader, states, references, epochs=1000, lr=0.001)

# Save model
torch.save(trained_model.state_dict(), 'final_mpc_dnn_model.pth')
print("\n[6/7] Model saved as 'final_mpc_dnn_model.pth'")

# Test controller
print("[7/7] Testing DNN controller on multiple initial conditions...")
dnn_controller = DNNController(trained_model, collector, t_param)

test_initial_states = [
    np.array([0.3, 0.0, np.pi/3]),
    np.array([0.2, 0.5, 0.0]),
    np.array([-0.15, -0.3, -np.pi/4]),
    np.array([0.0, 1.0, np.pi/2]),
    np.array([0.15, -0.4, -np.pi/6]),
]

trajectories = []
controls_list = []

for i, init_state in enumerate(test_initial_states):
    traj, controls = simulate_trajectory(dnn_controller, init_state, ts, steps=300)
    trajectories.append(traj)
    controls_list.append(controls)

plot_comprehensive_results(trajectories, controls_list, x_ref_array, y_ref_array, test_initial_states)

print("\n" + "="*70)
print(" ✓ Training and Evaluation Complete!")
print("="*70)
print("\nGenerated files:")
print("  • mpc_dataset.npz                 (Training data)")
print("  • best_mpc_dnn_model.pth          (Best model during training)")
print("  • final_mpc_dnn_model.pth         (Final model)")
print("  • training_curves.png             (Training progress)")
print("  • dnn_results_comprehensive.png   (Performance analysis)")
print("="*70 + "\n")