"""
LoRA implementation for Stable Diffusion UNet.
Based on PEFT library for efficient fine-tuning.
"""

from typing import Optional
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType


def setup_lora_config(
    rank: int = 4,
    alpha: int = 32,
    target_modules: Optional[list] = None,
    dropout: float = 0.0
) -> LoraConfig:
    """
    Setup LoRA configuration for Stable Diffusion UNet.
    
    Args:
        rank: LoRA rank (dimension of low-rank matrices)
        alpha: LoRA alpha (scaling factor)
        target_modules: List of module names to apply LoRA (default: attention layers)
        dropout: Dropout probability
    
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0"
        ]
    
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    return config


def apply_lora_to_unet(
    unet,
    config: LoraConfig,
    adapter_name: str = "default"
):
    """
    Apply LoRA to UNet model.
    
    Args:
        unet: UNet model from diffusers
        config: LoRA configuration
        adapter_name: Name of the adapter
    
    Returns:
        PEFT model with LoRA applied
    """
    model = get_peft_model(unet, config, adapter_name=adapter_name)
    return model


def load_lora_weights(model, checkpoint_path: str, adapter_name: str = "default"):
    """
    Load LoRA weights from checkpoint.
    
    Args:
        model: PEFT model
        checkpoint_path: Path to LoRA checkpoint
        adapter_name: Name of the adapter
    """
    model.load_adapter(checkpoint_path, adapter_name=adapter_name)
    return model

