import numpy as np
import time
from scipy.linalg import solve_discrete_are, inv

# 1. Precompute K once using discrete-time LQR
Q = np.diag([0.8, 4, 1.5])
R = np.diag([1, 1.8])
v_nom = 0.2
k_nom = 0.0
dt = 0.03  # Time step

# Discrete-time system matrices
A = np.array([[1, v_nom*k_nom*dt, -v_nom*dt], [-v_nom*k_nom*dt, 1, 0], [0, 0, 1]])
B = np.array([[dt, 0], [0, v_nom*dt], [0, dt]])

# Discrete LQR: K = (R + B^T P B)^-1 B^T P A
P = solve_discrete_are(A, B, Q, R)
K = inv(R + B.T @ P @ B) @ B.T @ P @ A

# 2. Simulate the execution time of one full lap (approx 1300 steps)
num_steps = 1300
v_ref = 0.2
k_path = 0.1
dummy_state = [0.1, 0.1, 0.1]
dummy_path = [0.12, 0.12, 0.12]

# Start the timer!
start_time = time.perf_counter()

for _ in range(num_steps):
    # Calculate error
    e = np.array([
        (dummy_state[0]-dummy_path[0])*np.cos(dummy_path[2]) + (dummy_state[1]-dummy_path[1])*np.sin(dummy_path[2]), 
        -(dummy_state[0]-dummy_path[0])*np.sin(dummy_path[2]) + (dummy_state[1]-dummy_path[1])*np.cos(dummy_path[2]), 
        dummy_state[2]-dummy_path[2]
    ])
    
    # The actual control step
    u = -K @ e 
    
    # Clip outputs
    v_out = np.clip(v_ref + u[0], 0.01, 0.4)
    w_out = np.clip(v_ref * k_path + u[1], -4, 4)

# Stop the timer!
end_time = time.perf_counter()

total_runtime = end_time - start_time
print(f"Total LQR runtime for {num_steps} steps: {total_runtime:.5f} seconds")