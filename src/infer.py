"""
Inference script for style transfer using trained LoRA.
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import yaml


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Inference with LoRA style transfer")
    parser.add_argument("--config", type=str, default="src/configs/lora_config.yaml")
    parser.add_argument("--content_image", type=str, required=True,
                       help="Path to content image")
    parser.add_argument("--lora_path", type=str, required=True,
                       help="Path to LoRA checkpoint")
    parser.add_argument("--output_path", type=str, default="output.png",
                       help="Path to save output image")
    parser.add_argument("--style_strength", type=float, default=1.0,
                       help="Strength of style transfer (0.0 to 1.0)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print(f"Loading base model: {config['base_model']}")
    pipe = StableDiffusionPipeline.from_pretrained(
        config['base_model'],
        torch_dtype=torch.float16
    )
    
    print(f"Loading LoRA from {args.lora_path}")
    pipe.unet.load_attn_procs(args.lora_path)
    
    content_image = Image.open(args.content_image).convert('RGB')
    content_image = content_image.resize((config['image_size'], config['image_size']))
    
    print("Generating styled image...")
    # Simplified inference - actual implementation needs proper pipeline
    # This is a placeholder structure
    
    output_image = pipe(
        prompt="",  # Empty prompt as per requirements
        image=content_image,
        num_inference_steps=50,
        guidance_scale=7.5,
        strength=args.style_strength
    ).images[0]
    
    output_image.save(args.output_path)
    print(f"Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

