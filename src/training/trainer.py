import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from ..controllers.dnn_controller import MPCNeuralNetwork
    from ..utils.visualization import plot_training_curves
except ImportError:
    try:
        from src.controllers.dnn_controller import MPCNeuralNetwork
        from src.utils.visualization import plot_training_curves
    except ImportError:
        from controllers.dnn_controller import MPCNeuralNetwork
        from utils.visualization import plot_training_curves


class MPCDataset(Dataset):
    """PyTorch dataset for MPC data."""
    
    def __init__(self, references, controls, standardize_targets=True):
        """
        Initialize dataset.
        
        Args:
            references: Array of reference features
            controls: Array of control actions
            standardize_targets: Whether to standardize control targets
        """
        self.references = torch.FloatTensor(references)
        
        # Extract only s and omega, drop v_path (which is nearly constant)
        self.controls = torch.FloatTensor(controls[:, :2])
        
        # Standardize targets (s and omega) to have zero mean and unit variance
        if standardize_targets:
            self.control_mean = self.controls.mean(dim=0)
            self.control_std = self.controls.std(dim=0) + 1e-8
            self.controls = (self.controls - self.control_mean) / self.control_std
        else:
            self.control_mean = None
            self.control_std = None
    
    def __len__(self):
        return len(self.references)
    
    def __getitem__(self, idx):
        return self.references[idx], self.controls[idx]


def train_model(model, train_loader, val_loader, references, epochs=1000, lr=0.001, seed=42, batch_size=32):
    """
    Train the neural network to mimic MPC.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        references: Training references for normalization
        epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed for reproducibility
        batch_size: Batch size (for GPU scaling)
    
    Returns:
        Trained model
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Limit CPU threads to reduce heat
    torch.set_num_threads(4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training neural network on {device} (seed={seed}, batch_size={batch_size})")
    
    # Fit normalization BEFORE moving to device to avoid device mismatch
    model.fit_normalization(references)
    model = model.to(device)
    
    # Scale learning rate with batch size
    scaled_lr = lr * (batch_size / 32)
    optimizer = optim.Adam(model.parameters(), lr=scaled_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=30, factor=0.5, verbose=False)
    criterion = nn.SmoothL1Loss()  # Huber loss - more robust to outliers than MSE
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 150
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for refs, controls in train_loader:
            refs, controls = refs.to(device), controls.to(device)
            predictions = model(refs)
            
            optimizer.zero_grad()
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
            for refs, controls in val_loader:
                refs, controls = refs.to(device), controls.to(device)
                predictions = model(refs)
                loss = criterion(predictions, controls)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_mpc_dnn_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('models/best_mpc_dnn_model.pth'))
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    return model


def prepare_dataloaders(references, controls, train_split=0.8, batch_size=32, block_ids=None, standardize_targets=True):
    """
    Prepare training and validation data loaders.
    
    Args:
        references: Array of reference features
        controls: Array of controls
        train_split: Fraction of data for training
        batch_size: Batch size for training
        block_ids: Optional array of block IDs for block-based validation split
        standardize_targets: Whether to standardize control targets
    
    Returns:
        train_loader, val_loader, dataset
    """
    dataset = MPCDataset(references, controls, standardize_targets=standardize_targets)
    
    if block_ids is not None:
        # Block-based validation split: split by anchor blocks
        unique_blocks = np.unique(block_ids)
        n_train_blocks = int(train_split * len(unique_blocks))
        
        # Randomly select blocks for training
        np.random.shuffle(unique_blocks)
        train_blocks = unique_blocks[:n_train_blocks]
        val_blocks = unique_blocks[n_train_blocks:]
        
        # Create masks for train/val based on block membership
        train_mask = np.isin(block_ids, train_blocks)
        val_mask = np.isin(block_ids, val_blocks)
        
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        print(f"Block-based split: {len(train_blocks)} training blocks, {len(val_blocks)} validation blocks")
        print(f"  - Training samples: {len(train_indices)}")
        print(f"  - Validation samples: {len(val_indices)}")
    else:
        # Random split (legacy)
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    
    return train_loader, val_loader, dataset
