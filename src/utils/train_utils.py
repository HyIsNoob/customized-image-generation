"""
Training utilities: save/load checkpoints, logging, etc.
"""

import torch
import os
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth"
):
    """
    Save training checkpoint.
    """
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None
):
    """
    Load training checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    return epoch, loss

