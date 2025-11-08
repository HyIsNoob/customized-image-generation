"""
Training script for LoRA fine-tuning on Stable Diffusion.
"""

import argparse
import yaml
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import DDPMScheduler
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

from src.models.lora import setup_lora_config
from src.utils.data_utils import create_dataloader
from src.utils.train_utils import save_checkpoint
from src.utils.eval_utils import compute_style_loss, compute_content_loss


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train LoRA for Stable Diffusion")
    parser.add_argument("--config", type=str, default="src/configs/lora_config.yaml",
                       help="Path to config file")
    parser.add_argument("--style_name", type=str, required=True,
                       help="Name of the style (e.g., 'monet', 'ukiyo-e')")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "fp16"),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1)
    )
    
    print(f"Loading base model: {config['base_model']}")
    pipe = StableDiffusionPipeline.from_pretrained(
        config['base_model'],
        torch_dtype=torch.float16 if config.get("mixed_precision") == "fp16" else torch.float32
    )
    
    unet = pipe.unet
    
    lora_config = setup_lora_config(
        rank=config['lora_rank'],
        alpha=config['lora_alpha'],
        target_modules=config['target_modules'],
        dropout=config['lora_dropout']
    )
    
    unet = get_peft_model(unet, lora_config)
    
    print(f"Trainable parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
    
    dataloader = create_dataloader(
        content_dir=config['content_dir'],
        style_dir=config['style_dir'],
        batch_size=config['batch_size'],
        image_size=config['image_size']
    )
    
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config['learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay'],
        eps=config['adam_epsilon']
    )
    
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    
    print("Starting training...")
    global_step = 0
    
    for epoch in range(config['num_epochs']):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                content_images = batch['content']
                style_images = batch['style']
                
                # Training step here
                # This is a simplified version - actual implementation needs
                # VAE encoding, noise scheduling, etc.
                
                if global_step % config['save_steps'] == 0:
                    save_path = f"{config['output_dir']}/{args.style_name}/checkpoint-{global_step}"
                    unet.save_pretrained(save_path)
                    print(f"Saved checkpoint at step {global_step}")
                
                global_step += 1
                
                if global_step >= config['max_train_steps']:
                    break
        
        if global_step >= config['max_train_steps']:
            break
    
    print("Training completed!")
    final_save_path = f"{config['output_dir']}/{args.style_name}/final"
    unet.save_pretrained(final_save_path)
    print(f"Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()

