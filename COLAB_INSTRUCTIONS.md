# Google Colab Training Instructions

## Overview
Use Google Colab's free GPU to train your DNN model, avoiding overheating your laptop.

## Step 1: Collect Data on Your Laptop

Run the data collection script on your laptop (CPU-only):

```bash
python scripts/collect_data_only.py
```

This will:
- Collect ~17k MPC samples with heading errors
- Save to `data/mpc_dataset.npz`
- Take ~5-10 minutes (less heat than full training)

## Step 2: Upload to Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Create a new notebook
3. Set Runtime → Change runtime type → **GPU** (T4)
4. Upload these files:
   - `mpc_dataset.npz` (from step 1)
   - `src/controllers/dnn_controller.py`
   - `src/utils/reference.py`
   - `src/utils/visualization.py`
   - `src/training/trainer.py`
   - `colab_train.py`

**Alternative**: Upload `colab_train.py` and run the file upload cell in the notebook.

## Step 3: Run Training

Option A: Using the Python script
```python
!python colab_train.py
```

Option B: Using the notebook
```python
# Run all cells in colab_training.ipynb
```

The training will:
- Use GPU (10-20x faster than CPU)
- Take ~10-20 minutes (vs hours on laptop)
- Save model to `models/trained_model_colab.pth`

## Step 4: Download Model

Download `models/trained_model_colab.pth` and save it to your laptop's `models/` folder.

## Step 5: Test on Your Laptop

Now test the trained model on your laptop using the evaluation script:

```bash
python scripts/test_dnn_on_ellipse.py
```

## Important Notes

### What the GPU Speeds Up
- ✅ Neural network training (matrix operations)
- ✅ Forward/backward passes
- ✅ Large batch sizes (256 instead of 32)

### What the GPU Does NOT Speed Up
- ❌ MPC data collection (Rust solver, CPU-only)
- ❌ DAgger on-policy labeling (requires MPC solves)
- ❌ Evaluation/testing (runs on CPU)

### Session Timeouts
- Colab sessions timeout after ~90 minutes of inactivity
- Sessions can be disconnected but keep running
- **Save checkpoints** if training longer than 60 minutes
- Download the model immediately when training completes

### Free Tier Limits
- Colab Free: ~30 hours GPU per week
- Kaggle: ~30 hours GPU per week (more stable)
- Use GPU only for training, not data collection

## Modified Code Changes

The following changes were made for GPU compatibility:

### 1. Batch Size Scaling
```python
# Automatic batch size selection
batch_size = 256 if torch.cuda.is_available() else 32

# Learning rate scales with batch size
scaled_lr = lr * (batch_size / 32)
```

### 2. Device Mismatch Fix
```python
# Fit normalization BEFORE moving to device
model.fit_normalization(references)
model = model.to(device)  # Move after normalization
```

### 3. CPU Thread Limiting
```python
# Reduce CPU threads to prevent overheating during data loading
torch.set_num_threads(4)
```

## File Structure for Colab

```
colab/
├── mpc_dataset.npz              # Upload from laptop
├── src/
│   ├── controllers/
│   │   └── dnn_controller.py    # Upload from laptop
│   ├── utils/
│   │   ├── reference.py         # Upload from laptop
│   │   └── visualization.py     # Upload from laptop
│   └── training/
│       └── trainer.py           # Upload from laptop
├── colab_train.py               # Upload this script
└── models/
    └── trained_model_colab.pth  # Download this after training
```

## Troubleshooting

### "ModuleNotFoundError"
- Make sure all source files are uploaded
- Check that `__init__.py` files exist in each src folder

### "CUDA out of memory"
- Reduce batch size: `batch_size = 128` or `64`
- Reduce network size: `hidden_dims=[512, 512, 256, 128]`

### Session disconnected
- Check if model was saved before disconnect
- If yes, download from files
- If no, restart training from scratch

### Training very slow
- Verify GPU is enabled: Runtime → Change runtime type → GPU
- Check device output: should say "Using: cuda"

## Alternative: Kaggle Notebooks

If Colab is unstable, try Kaggle:
1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Create new notebook
3. Settings → Accelerator → GPU
4. Upload files and run same script
5. More stable for long runs (up to 30 hours/week)
