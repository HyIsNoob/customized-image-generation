from typing import Dict, Tuple, Optional, List

import glob
import os
import random
import shutil

from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


def sample_style_images(source_dir: str, destination_dir: str, max_per_style: Optional[int] = None, split: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 2025) -> Dict[str, int]:
    random.seed(seed)
    os.makedirs(destination_dir, exist_ok=True)
    stats: Dict[str, int] = {}
    for subset in ("train", "valid", "test"):
        os.makedirs(os.path.join(destination_dir, subset), exist_ok=True)
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    for style in sorted(d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))):
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(source_dir, style, f"*{ext}")))
        if not image_paths:
            continue
        if max_per_style is not None:
            image_paths = random.sample(image_paths, min(len(image_paths), max_per_style))
        n_total = len(image_paths)
        n_train = int(n_total * split[0])
        n_valid = int(n_total * split[1])
        assignments = {
            "train": image_paths[:n_train],
            "valid": image_paths[n_train:n_train + n_valid],
            "test": image_paths[n_train + n_valid:]
        }
        for subset, paths in assignments.items():
            for idx, path in enumerate(paths, 1):
                dest_name = f"{style}_{idx:03d}{os.path.splitext(path)[1].lower()}"
                shutil.copy(path, os.path.join(destination_dir, subset, dest_name))
        stats[style] = n_total
    return stats


class StableDiffusionTransform:
    def __init__(self, size: int = 512, center_crop: bool = True):
        ops: List[T.transforms.Compose] = []
        ops.append(T.Resize(size, interpolation=T.InterpolationMode.BILINEAR))
        if center_crop:
            ops.append(T.CenterCrop(size))
        ops.append(T.ToTensor())
        ops.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transform = T.Compose(ops)

    def __call__(self, image: Image.Image):
        return self.transform(image)


class CaptionedImageDataset(Dataset):
    def __init__(self, data_dir: str, valid_ext: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"), transform: Optional[StableDiffusionTransform] = None):
        self.image_paths = []
        for ext in valid_ext:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
        if not self.image_paths:
            raise RuntimeError(f"No images found in {data_dir}")
        self.image_paths = sorted(self.image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        caption_path = os.path.splitext(path)[0] + ".txt"
        caption = ""
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as handle:
                caption = handle.read().strip()
        return {
            "image": image,
            "caption": caption,
            "path": path
        }


class ContentStylePairDataset(Dataset):
    def __init__(self, content_dir: str, style_dir: str, transform: Optional[StableDiffusionTransform] = None, style_transform: Optional[StableDiffusionTransform] = None, valid_ext: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
        self.content_paths = []
        self.style_paths = []
        for ext in valid_ext:
            self.content_paths.extend(glob.glob(os.path.join(content_dir, f"*{ext}")))
            self.style_paths.extend(glob.glob(os.path.join(style_dir, f"*{ext}")))
        if not self.content_paths:
            raise RuntimeError(f"No content images found in {content_dir}")
        if not self.style_paths:
            raise RuntimeError(f"No style images found in {style_dir}")
        self.transform = transform
        self.style_transform = style_transform or transform

    def __len__(self) -> int:
        return len(self.content_paths)

    def __getitem__(self, index: int):
        content_path = self.content_paths[index % len(self.content_paths)]
        style_path = random.choice(self.style_paths)
        content_image = Image.open(content_path).convert("RGB")
        style_image = Image.open(style_path).convert("RGB")
        if self.transform is not None:
            content_image = self.transform(content_image)
        if self.style_transform is not None:
            style_image = self.style_transform(style_image)
        return {
            "content": content_image,
            "style": style_image,
            "content_path": content_path,
            "style_path": style_path
        }


def create_lora_dataloader(style_dir: str, batch_size: int = 4, num_workers: int = 2, image_size: int = 512, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
    transform = StableDiffusionTransform(size=image_size)
    dataset = CaptionedImageDataset(style_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last)


def create_content_style_dataloader(content_dir: str, style_dir: str, batch_size: int = 4, num_workers: int = 2, image_size: int = 512, shuffle: bool = True) -> DataLoader:
    transform = StableDiffusionTransform(size=image_size)
    dataset = ContentStylePairDataset(content_dir, style_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def create_dreambooth_dataloaders(instance_dir: str, class_dir: Optional[str] = None, batch_size: int = 1, num_workers: int = 2, image_size: int = 512, shuffle: bool = True) -> Dict[str, DataLoader]:
    transform = StableDiffusionTransform(size=image_size)
    instance_dataset = CaptionedImageDataset(instance_dir, transform=transform)
    loaders: Dict[str, DataLoader] = {
        "instance": DataLoader(instance_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
    }
    if class_dir is not None:
        class_dataset = CaptionedImageDataset(class_dir, transform=transform)
        loaders["class"] = DataLoader(class_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
    return loaders
