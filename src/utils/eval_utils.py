from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms as T
from torchvision.models import vgg19, VGG19_Weights

try:
    import lpips
except ImportError as exc:
    raise ImportError("lpips package is required. Install with `pip install lpips`.") from exc

__all__ = [
    "FIDEvaluator",
    "VGGFeatureExtractor",
    "compute_content_loss",
    "compute_style_loss",
    "compute_lpips",
    "compute_ssim",
    "prepare_for_vgg",
    "sd_to_zero_one",
]

_VGG_MEAN = (0.485, 0.456, 0.406)
_VGG_STD = (0.229, 0.224, 0.225)
_VGG_NORMALIZE = T.Normalize(mean=_VGG_MEAN, std=_VGG_STD)

_LAYER_NAMES = [
    "conv1_1",
    "relu1_1",
    "conv1_2",
    "relu1_2",
    "pool1",
    "conv2_1",
    "relu2_1",
    "conv2_2",
    "relu2_2",
    "pool2",
    "conv3_1",
    "relu3_1",
    "conv3_2",
    "relu3_2",
    "conv3_3",
    "relu3_3",
    "conv3_4",
    "relu3_4",
    "pool3",
    "conv4_1",
    "relu4_1",
    "conv4_2",
    "relu4_2",
    "conv4_3",
    "relu4_3",
    "conv4_4",
    "relu4_4",
    "pool4",
    "conv5_1",
    "relu5_1",
    "conv5_2",
    "relu5_2",
    "conv5_3",
    "relu5_3",
    "conv5_4",
    "relu5_4",
    "pool5",
]


def sd_to_zero_one(images: Tensor) -> Tensor:
    return (images * 0.5 + 0.5).clamp(0.0, 1.0)


def prepare_for_vgg(images: Tensor) -> Tensor:
    return _VGG_NORMALIZE(sd_to_zero_one(images))


def compute_gram_matrix(features: Tensor) -> Tensor:
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size, channels, height * width)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (channels * height * width)


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers: Iterable[str], device: torch.device | str | None = None):
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1
        self.model = vgg19(weights=weights).features
        for param in self.model.parameters():
            param.requires_grad = False
        self.target_layers = set(layers)
        if device is not None:
            self.to(device)
        self.eval()

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        features: Dict[str, Tensor] = {}
        x = images
        for idx, layer in enumerate(self.model):
            x = layer(x)
            layer_name = _LAYER_NAMES[idx]
            if layer_name in self.target_layers:
                features[layer_name] = x
            if len(features) == len(self.target_layers):
                break
        return features


def _ensure_extractor(
    extractor: VGGFeatureExtractor | None,
    layers: Sequence[str],
    device: torch.device | str,
) -> VGGFeatureExtractor:
    if extractor is None:
        return VGGFeatureExtractor(layers=layers, device=device)
    if not set(layers).issubset(extractor.target_layers):
        raise ValueError("Provided extractor does not cover all requested layers.")
    return extractor.to(device)


def compute_content_loss(
    output: Tensor,
    content: Tensor,
    extractor: VGGFeatureExtractor | None = None,
    layers: Sequence[str] = ("relu4_2",),
) -> Tensor:
    device = output.device
    extractor = _ensure_extractor(extractor, layers, device)
    output_features = extractor(prepare_for_vgg(output))
    content_features = extractor(prepare_for_vgg(content))
    loss = torch.zeros(1, device=device, dtype=output.dtype)
    for layer in layers:
        loss = loss + F.mse_loss(output_features[layer], content_features[layer])
    return loss / len(layers)


def compute_style_loss(
    output: Tensor,
    style: Tensor,
    extractor: VGGFeatureExtractor | None = None,
    layers: Sequence[str] = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"),
) -> Tensor:
    device = output.device
    extractor = _ensure_extractor(extractor, layers, device)
    output_features = extractor(prepare_for_vgg(output))
    style_features = extractor(prepare_for_vgg(style))
    loss = torch.zeros(1, device=device, dtype=output.dtype)
    for layer in layers:
        gram_output = compute_gram_matrix(output_features[layer])
        gram_style = compute_gram_matrix(style_features[layer])
        loss = loss + F.mse_loss(gram_output, gram_style)
    return loss / len(layers)


def compute_lpips(
    output: Tensor,
    target: Tensor,
    model: lpips.LPIPS | None = None,
) -> Tensor:
    device = output.device
    lpips_model = model if model is not None else lpips.LPIPS(net="vgg")
    lpips_model = lpips_model.to(device)
    lpips_model.eval()
    with torch.no_grad():
        score = lpips_model(output, target)
    return score.mean()


def compute_ssim(
    output: Tensor,
    target: Tensor,
) -> Tensor:
    output_01 = sd_to_zero_one(output).clamp(0.0, 1.0)
    target_01 = sd_to_zero_one(target).clamp(0.0, 1.0)
    return structural_similarity_index_measure(output_01, target_01, data_range=1.0)


class FIDEvaluator:
    def __init__(self, device: torch.device | str = "cpu"):
        self.metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    @property
    def device(self) -> torch.device:
        return next(self.metric.parameters()).device

    def reset(self) -> None:
        self.metric.reset()

    def update_real(self, images: Tensor) -> None:
        images_01 = sd_to_zero_one(images).clamp(0.0, 1.0)
        self.metric.update(images_01.to(self.device), real=True)

    def update_fake(self, images: Tensor) -> None:
        images_01 = sd_to_zero_one(images).clamp(0.0, 1.0)
        self.metric.update(images_01.to(self.device), real=False)

    def compute(self) -> float:
        return float(self.metric.compute())

