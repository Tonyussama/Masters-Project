import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from ..controllers.dnn_controller import MPCNeuralNetwork
    from ..utils.visualization import plot_training_curves
except ImportError:
    from src.controllers.dnn_controller import MPCNeuralNetwork
    from src.utils.visualization import plot_training_curves


class MPCDataset(Dataset):
    """PyTorch dataset for MPC data."""
    
    def __init__(self, states, references, controls):
        """
        Initialize dataset.
        
        Args:
            states: Array of geometric states
            references: Array of reference features
            controls: Array of control actions
        """
        self.states = torch.FloatTensor(states)
        self.references = torch.FloatTensor(references)
        self.controls = torch.FloatTensor(controls)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.references[idx], self.controls[idx]


def train_model(model, train_loader, val_loader, states, references, epochs=300, lr=0.001):
    """
    Train the neural network to mimic MPC.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        states: Training states for normalization
        references: Training references for normalization
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained model
    """
    model.fit_normalization(states, references)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training neural network on {device}")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     patience=15, factor=0.5, verbose=False)
    criterion = nn.MSELoss()
    
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


def prepare_dataloaders(states, references, controls, train_split=0.8, batch_size=32):
    """
    Prepare training and validation data loaders.
    
    Args:
        states: Array of states
        references: Array of reference features
        controls: Array of controls
        train_split: Fraction of data for training
        batch_size: Batch size for training
    
    Returns:
        train_loader, val_loader, dataset
    """
    dataset = MPCDataset(states, references, controls)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    
    return train_loader, val_loader, dataset
