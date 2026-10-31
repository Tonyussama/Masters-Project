# DNN Improvements Implementation Summary

## Overview
This document summarizes the advanced improvements made to the DNN training pipeline to achieve better performance and address error spikes.

## Implemented Improvements

### 1. ✅ DAgger (Dataset Aggregation) for On-Policy Data Collection
**File:** `scripts/train_dagger.py`

Implements iterative dataset augmentation:
- **Round 0:** Initial MPC dataset with heading errors
- **Round 1-3:** Run DNN in closed-loop, collect states it visits, label with MPC, add to dataset
- **Disturbances:** Includes bias (persistent) and impulse (single-step) disturbances for robustness
- **Result:** Dataset grows with DNN's actual experience, reducing distribution shift

**Usage:**
```bash
python scripts/train_dagger.py
```

### 2. ✅ Heading Errors in Training Seeds
**File:** `src/utils/data_collection.py`

Previously, seeds had heading aligned to path tangent. Now:
- Adds random heading errors uniformly sampled from `[-π/6, +π/6]` (±30°)
- Network now learns to handle heading misalignment
- **Parameter:** `heading_error_range` in `collect_data_corridor()`

### 3. ✅ Keep/Log MPC Failure Cases
**File:** `src/utils/data_collection.py`

Previously, MPC failures were discarded. Now:
- Returns `failure_info` dictionary with failure statistics
- Logs exit status, heading error, and state for each failure
- **Purpose:** These are the hardest states - valuable for analysis
- **Parameter:** `keep_failures=True` in `collect_data_corridor()`

### 4. ✅ Remove Global State Inputs
**File:** `src/controllers/dnn_controller.py`

Previously, DNN used global [x, y, θ] + robot-frame features. Now:
- **Inputs:** Only robot-frame reference features (42 dims)
- **Rationale:** Path preview already encodes pose relative to path
- **Benefit:** Makes DNN generalizable to any reference path (not just this ellipse)
- **Architecture change:** `MPCNeuralNetwork` no longer takes state input

### 5. ✅ Block-Based Validation Split
**Files:** `src/utils/data_collection.py`, `src/training/trainer.py`

Previously, random split put neighboring samples in both sets (optimistic validation). Now:
- Splits by anchor blocks (each anchor = 9 samples)
- Training/validation samples are from different path regions
- **More realistic:** Validation tests generalization, not memorization
- **Parameter:** `block_ids` returned by `collect_data_corridor()`

### 6. ✅ Standardize Targets and Drop v_path
**Files:** `src/controllers/dnn_controller.py`, `src/training/trainer.py`

Previously, used raw controls [s, ω, v_path] with MSE loss. Now:
- **Outputs:** Only [s, ω] (dropped v_path)
- **v_path:** Set to constant v_ref in controller
- **Standardization:** Targets standardized to zero mean, unit variance
- **Loss:** Changed from MSE to SmoothL1 (Huber loss) - more robust to outliers
- **Benefit:** ω no longer dominates the loss; balanced learning

### 7. ✅ MPC Fallback for Unsafe Outputs
**File:** `src/controllers/dnn_controller.py`

Added safety mechanism to DNN controller:
- **Threshold:** Cross-track error > 0.5m triggers MPC fallback
- **Fallback:** Uses MPC to recover when DNN is uncertain
- **Statistics:** Tracks fallback rate (`get_fallback_stats()`)
- **Purpose:** Addresses "what about guarantees?" question
- **Usage:**
```python
dnn_controller = DNNController(
    model, collector, t_param, v_ref=v_ref,
    mpc_solver=solver,  # Optional: None = no fallback
    cross_track_threshold=0.5
)
```

## Configuration Changes

### Updated Training Parameters
```python
# Data collection
num_anchors = 5000  # (was 1000)
offset_distance = 0.1
heading_error_range = np.pi/6  # NEW
keep_failures = True  # NEW

# Network architecture
hidden_dims = [1024, 1024, 512, 256, 128]  # (was [128, 128, 64, 32])
output_dim = 2  # (was 3) - dropped v_path

# Training
epochs = 1000  # (was 300)
early_stop_patience = 40
loss = nn.SmoothL1Loss()  # (was MSE)
seed = 42  # NEW: fixed for reproducibility
```

## New Scripts

### `scripts/train_dagger.py`
Full DAgger implementation with:
- 3 rounds of iterative data collection
- Bias and impulse disturbances
- On-policy data labeling with MPC
- Final training on augmented dataset

### Updated `scripts/train_mpc_to_dnn.py`
Now uses:
- Heading errors in seeds
- Block-based validation split
- Standardized targets
- Robot-frame-only inputs
- Tracking error metrics (RMS, max)

## Migration Guide

### For Existing Training Pipelines
If you have existing code that calls `collect_data_corridor()`:

**Old:**
```python
states, references, controls = collector.collect_data_corridor(
    num_anchors=1000, offset_distance=0.1
)
```

**New:**
```python
states, references, controls, failure_info, block_ids = collector.collect_data_corridor(
    num_anchors=5000, offset_distance=0.1,
    heading_error_range=np.pi/6, keep_failures=True
)
```

### For Training
**Old:**
```python
train_loader, val_loader, dataset = prepare_dataloaders(
    states, references, controls, train_split=0.8, batch_size=32
)
model = MPCNeuralNetwork(state_dim=3, ref_feature_dim=ref_feature_dim, 
                         hidden_dims=[128, 128, 64, 32], output_dim=3)
trained_model = train_model(model, train_loader, val_loader, states, references, 
                            epochs=300, lr=0.001)
```

**New:**
```python
train_loader, val_loader, dataset = prepare_dataloaders(
    states, references, controls, train_split=0.8, batch_size=32,
    block_ids=block_ids, standardize_targets=True
)
model = MPCNeuralNetwork(ref_feature_dim=ref_feature_dim, 
                        hidden_dims=[1024, 1024, 512, 256, 128], output_dim=2)
trained_model = train_model(model, train_loader, val_loader, references, 
                            epochs=1000, lr=0.001, seed=42)
```

### For Inference
**Old:**
```python
dnn_controller = DNNController(model, collector, t_param)
```

**New:**
```python
dnn_controller = DNNController(model, collector, t_param, v_ref=v_ref)
# With MPC fallback:
dnn_controller = DNNController(model, collector, t_param, v_ref=v_ref,
                             mpc_solver=solver, cross_track_threshold=0.5)
```

## Expected Improvements

1. **Reduced Error Spikes:** DAgger targets distribution shift error
2. **Better Generalization:** Heading errors and robot-frame inputs
3. **More Robust Validation:** Block-based split prevents overfitting
4. **Balanced Learning:** Standardized targets and dropped v_path
5. **Safety Guarantees:** MPC fallback for recovery
6. **Path Independence:** Robot-frame inputs enable any reference path

## Next Steps

1. **Run DAgger training:**
   ```bash
   python scripts/train_dagger.py
   ```

2. **Evaluate performance:**
   - Compare RMS and max error before/after
   - Check fallback rate with MPC fallback enabled

3. **Multiple runs for statistics:**
   - Run with different seeds (42, 123, 456)
   - Report mean ± std of performance metrics

4. **Document in thesis:**
   - Add section on DAgger methodology
   - Explain robot-frame inputs for generalization
   - Discuss MPC fallback as safety mechanism
   - Report failure analysis from MPC failure logging

## Files Modified

- `src/utils/data_collection.py` - Heading errors, failure logging, block IDs
- `src/controllers/dnn_controller.py` - Robot-frame inputs, MPC fallback
- `src/training/trainer.py` - Block-based split, standardization, SmoothL1 loss
- `scripts/train_mpc_to_dnn.py` - Updated to use new features
- `scripts/train_dagger.py` - NEW: DAgger implementation

## Notes

- The old dataset format is **not** compatible with the new code
- You must re-collect data with the updated `collect_data_corridor()`
- Training time will be longer due to:
  - More data (5000 anchors vs 1000)
  - Larger network (1024x1024x512x256x128 vs 128x128x64x32)
  - More epochs (1000 vs 300)
- DAgger adds additional time for iterative data collection
