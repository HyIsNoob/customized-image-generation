"""
Data utilities for loading and preprocessing images.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import Optional


class StyleTransferDataset(Dataset):
    """
    Dataset for style transfer training.
    Pairs content images with style images.
    """
    
    def __init__(
        self,
        content_dir: str,
        style_dir: str,
        pairs_file: Optional[str] = None,
        image_size: int = 512,
        augment: bool = True
    ):
        """
        Args:
            content_dir: Directory containing content images
            style_dir: Directory containing style images
            pairs_file: Optional CSV file with content-style pairs
            image_size: Target image size
            augment: Whether to apply data augmentation
        """
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.image_size = image_size
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        
        if pairs_file and os.path.exists(pairs_file):
            import pandas as pd
            self.pairs = pd.read_csv(pairs_file)
        else:
            self.pairs = None
    
    def __len__(self):
        if self.pairs is not None:
            return len(self.pairs)
        return len(os.listdir(self.content_dir))
    
    def __getitem__(self, idx):
        if self.pairs is not None:
            content_path = os.path.join(self.content_dir, self.pairs.iloc[idx]['content'])
            style_path = os.path.join(self.style_dir, self.pairs.iloc[idx]['style'])
        else:
            content_files = os.listdir(self.content_dir)
            style_files = os.listdir(self.style_dir)
            content_path = os.path.join(self.content_dir, content_files[idx % len(content_files)])
            style_path = os.path.join(self.style_dir, style_files[idx % len(style_files)])
        
        content_img = Image.open(content_path).convert('RGB')
        style_img = Image.open(style_path).convert('RGB')
        
        content_tensor = self.transform(content_img)
        style_tensor = self.transform(style_img)
        
        return {
            'content': content_tensor,
            'style': style_tensor
        }


def create_dataloader(
    content_dir: str,
    style_dir: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader for style transfer dataset.
    """
    dataset = StyleTransferDataset(content_dir, style_dir, **kwargs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

