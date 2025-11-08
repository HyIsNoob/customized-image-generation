# Implementation chi tiết sẽ được thêm sau khi có trained models và output images.


import torch
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
import numpy as np
from scipy import linalg


def compute_gram_matrix(features):
    """
    Compute Gram matrix for style loss.
    """
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size, channels, height * width)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (channels * height * width)


def compute_style_loss(output_features, style_features):
    """
    Compute style loss using Gram matrices.
    """
    output_gram = compute_gram_matrix(output_features)
    style_gram = compute_gram_matrix(style_features)
    return F.mse_loss(output_gram, style_gram)


def compute_content_loss(output_features, content_features):
    """
    Compute content loss using MSE on feature maps.
    """
    return F.mse_loss(output_features, content_features)


def compute_lpips(output_img, target_img, lpips_model):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).
    """
    return lpips_model(output_img, target_img).mean()


def compute_ssim(output_img, target_img):
    """
    Compute SSIM (Structural Similarity Index).
    Simplified version - for full implementation use scikit-image.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError("scikit-image is required for SSIM computation")
    
    output_np = output_img.squeeze().cpu().numpy().transpose(1, 2, 0)
    target_np = target_img.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    if output_np.shape[-1] == 1:
        output_np = output_np.squeeze(-1)
        target_np = target_np.squeeze(-1)
    
    return ssim(output_np, target_np, data_range=1.0, channel_axis=-1 if len(output_np.shape) == 3 else None)


def compute_fid(real_features, fake_features):
    """
    Compute FID (Fréchet Inception Distance).
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

