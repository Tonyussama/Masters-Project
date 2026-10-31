import numpy as np
import matplotlib.pyplot as plt

try:
    from .reference import find_projection_index
except ImportError:
    from src.utils.reference import find_projection_index


def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
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
    plt.savefig('outputs/training_curves.png', dpi=200, bbox_inches='tight')
    plt.show()


def plot_comprehensive_results(trajectories, controls_list, x_ref_array, y_ref_array, initial_states, ts):
    """Create comprehensive visualization of controller performance.

    Args:
        trajectories: List of trajectory arrays
        controls_list: List of control arrays
        x_ref_array: Reference x positions
        y_ref_array: Reference y positions
        initial_states: List of initial states
        ts: Time step
    """
    fig = plt.figure(figsize=(20, 14))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    # Plot 1: Full trajectories
    ax1 = plt.subplot(3, 3, 1)
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
    ax2 = plt.subplot(3, 3, 2)
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
    ax3 = plt.subplot(3, 3, 3)
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

    # Plot 4: Cross-track error along path
    ax4 = plt.subplot(3, 3, 4)
    for i, traj in enumerate(trajectories):
        errors = []
        path_params = []
        for state in traj:
            idx = find_projection_index(state, x_ref_array, y_ref_array)
            error = np.sqrt((state[0] - x_ref_array[idx])**2 +
                          (state[1] - y_ref_array[idx])**2)
            errors.append(error)
            path_params.append(state[3])  # theta_p (path parameter)
        plt.plot(path_params, errors, '-', color=colors[i], linewidth=2, alpha=0.8)
    plt.xlabel('Path Parameter [rad]', fontsize=12, fontweight='bold')
    plt.ylabel('Cross-track Error [m]', fontsize=12, fontweight='bold')
    plt.title('Cross-track Error Along Path', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 5: Linear velocity
    ax5 = plt.subplot(3, 3, 5)
    for i, controls in enumerate(controls_list):
        time = np.arange(len(controls)) * ts
        plt.plot(time, controls[:, 0], '-', color=colors[i], linewidth=2, alpha=0.8)
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Forward Speed (s) [m/s]', fontsize=12)
    plt.title('Control: Forward Speed', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 4.1])

    # Plot 6: Angular velocity
    ax6 = plt.subplot(3, 3, 6)
    for i, controls in enumerate(controls_list):
        time = np.arange(len(controls)) * ts
        plt.plot(time, controls[:, 1], '-', color=colors[i], linewidth=2, alpha=0.8)
    plt.xlabel('Time [s]', fontsize=12, fontweight='bold')
    plt.ylabel('Angular Velocity (ω) [rad/s]', fontsize=12)
    plt.title('Control: Angular Velocity', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 7: Path parameter velocity
    ax7 = plt.subplot(3, 3, 7)
    for i, controls in enumerate(controls_list):
        time = np.arange(len(controls)) * ts
        plt.plot(time, controls[:, 2], '-', color=colors[i], linewidth=2, alpha=0.8)
    plt.xlabel('Time [s]', fontsize=12, fontweight='bold')
    plt.ylabel('Path Velocity (v) [rad/s]', fontsize=12)
    plt.title('Control: Path Parameter Velocity', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 0.5])

    # Plot 8: Path parameter over time (progress along track)
    ax8 = plt.subplot(3, 3, 8)
    for i, traj in enumerate(trajectories):
        time = np.arange(len(traj)) * ts
        plt.plot(time, traj[:, 3], '-', color=colors[i], linewidth=2, alpha=0.8)
    plt.xlabel('Time [s]', fontsize=12, fontweight='bold')
    plt.ylabel('Path Parameter [rad]', fontsize=12, fontweight='bold')
    plt.title('Progress Along Track', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 9: Lap completion times
    ax9 = plt.subplot(3, 3, 9)
    lap_times_list = []
    for i, traj in enumerate(trajectories):
        path_params = traj[:, 3]
        # Detect lap completions (when path parameter wraps around)
        lap_start_time = 0
        lap_times = []
        for j in range(1, len(path_params)):
            # Check if we wrapped around (decrease in path parameter indicates lap completion)
            if path_params[j] < path_params[j-1]:
                lap_time = j * ts - lap_start_time
                lap_times.append(lap_time)
                lap_start_time = j * ts
        lap_times_list.append(lap_times)

        # Plot as points
        if lap_times:
            plt.plot(range(1, len(lap_times)+1), lap_times, 'o-',
                    color=colors[i], linewidth=2, markersize=8, alpha=0.8,
                    label=f'Init {i+1}')
            print(f"Trajectory {i+1}: Lap times = {[f'{t:.2f}s' for t in lap_times]}")

    plt.xlabel('Lap Number', fontsize=12, fontweight='bold')
    plt.ylabel('Lap Time [s]', fontsize=12, fontweight='bold')
    plt.title('Lap Completion Times', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    if any(lap_times_list):
        plt.legend(fontsize=9, loc='best')

    plt.suptitle('DNN Controller Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/dnn_results_comprehensive.png', dpi=200, bbox_inches='tight')
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

        # Print lap completion statistics
        if lap_times_list[i]:
            print(f"  Lap times:    {[f'{t:.2f}s' for t in lap_times_list[i]]}")
            print(f"  Avg lap time: {np.mean(lap_times_list[i]):.2f}s")
