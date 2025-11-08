# Visualization utilities for style transfer results.

import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid


def visualize_style_transfer(
    content_img,
    style_img,
    output_img,
    save_path: str = None,
    title: str = "Style Transfer"
):
    """
    Visualize style transfer results side-by-side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    images = [content_img, style_img, output_img]
    titles = ["Content", "Style", "Output"]
    
    for ax, img, t in zip(axes, images, titles):
        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img = np.clip((img + 1) / 2, 0, 1)
        
        ax.imshow(img)
        ax.set_title(t)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(
    losses: dict,
    save_path: str = None
):
    """
    Plot training loss curves.
    """
    fig, axes = plt.subplots(1, len(losses), figsize=(5*len(losses), 4))
    
    if len(losses) == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, losses.items()):
        ax.plot(values)
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

