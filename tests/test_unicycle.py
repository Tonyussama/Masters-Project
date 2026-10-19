import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unicycle import KinematicUnicycle


def test_unicycle():
    """Test the kinematic unicycle model."""
    
    # Initialize unicycle with default parameters
    unicycle = KinematicUnicycle()
    
    # Check initial state
    initial_state = unicycle.get_state()
    print(f"Initial state: x={initial_state[0]:.4f}, y={initial_state[1]:.4f}, theta={initial_state[2]:.4f}")
    assert np.allclose(initial_state, [0.1, 0.0, np.pi/2]), "Initial state mismatch"
    
    # Test forward motion (v=1.0, omega=0.0)
    u = np.array([1.0, 0.0])
    state = unicycle.step(u)
    print(f"After forward step: x={state[0]:.4f}, y={state[1]:.4f}, theta={state[2]:.4f}")
    
    # With theta=pi/2, the robot should move in positive y direction
    expected_x = 0.1 + 1.0 * np.cos(np.pi/2) * 0.03
    expected_y = 0.0 + 1.0 * np.sin(np.pi/2) * 0.03
    expected_theta = np.pi/2
    assert np.allclose(state, [expected_x, expected_y, expected_theta], atol=1e-10), "Forward motion mismatch"
    
    # Test rotation (v=0.0, omega=0.5)
    u = np.array([0.0, 0.5])
    state = unicycle.step(u)
    print(f"After rotation step: x={state[0]:.4f}, y={state[1]:.4f}, theta={state[2]:.4f}")
    
    expected_theta = np.pi/2 + 0.5 * 0.03
    assert np.allclose(state[0], expected_x, atol=1e-10), "x should not change during pure rotation"
    assert np.allclose(state[1], expected_y, atol=1e-10), "y should not change during pure rotation"
    assert np.allclose(state[2], expected_theta, atol=1e-10), "theta mismatch"
    
    # Test angle normalization (rotate by 2*pi)
    unicycle.reset()
    u = np.array([0.0, 2 * np.pi / 0.03])  # Rotate exactly 2*pi in one step
    state = unicycle.step(u)
    print(f"After 2pi rotation: x={state[0]:.4f}, y={state[1]:.4f}, theta={state[2]:.4f}")
    assert state[2] == np.pi/2, "Angle normalization failed"
    
    # Test angle normalization (rotate beyond pi)
    unicycle.reset()
    u = np.array([0.0, 2.0])  # Rotate by 2 rad
    state = unicycle.step(u)
    print(f"After +2 rad rotation: x={state[0]:.4f}, y={state[1]:.4f}, theta={state[2]:.4f}")
    # pi/2 + 2.0 = 3.5708 rad, which wraps to -2.7124 rad
    expected_theta = np.pi/2 + 2.0 * 0.03  # omega * dt
    print(f"Expected raw theta: {expected_theta:.4f}")
    normalized_theta = unicycle._normalize_angle(expected_theta)
    print(f"Normalized theta: {normalized_theta:.4f}")
    assert np.allclose(state[2], normalized_theta), "Angle normalization failed"
    
    # Test angle normalization (rotate beyond -pi)
    unicycle.reset()
    u = np.array([0.0, -4.0])  # Rotate by -4 rad/s
    state = unicycle.step(u)
    print(f"After -4 rad/s rotation: x={state[0]:.4f}, y={state[1]:.4f}, theta={state[2]:.4f}")
    expected_theta = np.pi/2 - 4.0 * 0.03  # omega * dt
    normalized_theta = unicycle._normalize_angle(expected_theta)
    assert np.allclose(state[2], normalized_theta), "Angle normalization failed"
    
    # Test combined motion
    unicycle.reset()
    u = np.array([1.0, 0.5])
    state = unicycle.step(u)
    print(f"After combined step: x={state[0]:.4f}, y={state[1]:.4f}, theta={state[2]:.4f}")
    
    # Multiple steps
    unicycle.reset()
    for i in range(10):
        u = np.array([0.5, 0.1])
        state = unicycle.step(u)
    print(f"After 10 steps: x={state[0]:.4f}, y={state[1]:.4f}, theta={state[2]:.4f}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_unicycle()
