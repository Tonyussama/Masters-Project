"""
DNN Training on Google Colab (GPU)

Instructions:
1. Upload these files to Colab:
   - mpc_dataset.npz (from running collect_data_only.py on your laptop)
   - src/controllers/dnn_controller.py
   - src/utils/reference.py  
   - src/utils/visualization.py
   - src/training/trainer.py
2. Set Runtime → Change runtime type → GPU
3. Run this script
4. Download the trained model
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Create directories
os.makedirs('src/controllers', exist_ok=True)
os.makedirs('src/utils', exist_ok=True)
os.makedirs('src/training', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Create __init__ files
for d in ['src', 'src/controllers', 'src/utils', 'src/training']:
    Path(f'{d}/__init__.py').touch()

print("="*70)
print(" DNN Training on Google Colab")
print("="*70)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Device] Using: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load dataset
print("\n[1/5] Loading dataset...")
data = np.load('mpc_dataset.npz')
states = data['states']
references = data['references']
controls = data['controls']
block_ids = data['block_ids']
print(f"  Samples: {len(states)}")
print(f"  Reference features: {references.shape[1]} dims")
print(f"  Controls: {controls.shape[1]} dims")

# Import modules
print("\n[2/5] Importing modules...")
sys.path.insert(0, 'src')
from controllers.dnn_controller import MPCNeuralNetwork
from training.trainer import prepare_dataloaders, train_model

# Prepare data loaders
print("\n[3/5] Preparing data loaders...")
batch_size = 256 if torch.cuda.is_available() else 32
print(f"  Batch size: {batch_size} ({'GPU' if torch.cuda.is_available() else 'CPU'})")

train_loader, val_loader, dataset = prepare_dataloaders(
    references, controls, train_split=0.8, batch_size=batch_size,
    block_ids=block_ids, standardize_targets=True
)
print(f"  Training: {len(train_loader.dataset)} samples")
print(f"  Validation: {len(val_loader.dataset)} samples")

# Create model
print("\n[4/5] Creating model...")
ref_feature_dim = references.shape[1]
model = MPCNeuralNetwork(
    ref_feature_dim=ref_feature_dim,
    hidden_dims=[1024, 1024, 512, 256, 128],
    output_dim=2
)
num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params:,}")

# Train
print("\n[5/5] Training model...")
trained_model = train_model(
    model, train_loader, val_loader, references,
    epochs=1000, lr=0.001, seed=42, batch_size=batch_size
)

# Save
torch.save(trained_model.state_dict(), 'models/trained_model_colab.pth')
print("\n" + "="*70)
print(" Training Complete!")
print("="*70)
print("\nModel saved as: models/trained_model_colab.pth")
print("Download this file to use on your laptop.")
print("="*70)
