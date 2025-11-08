"""
Gradio demo application for style transfer.
"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import yaml
from pathlib import Path


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_models(config):
    """Load base model and available LoRA checkpoints."""
    base_model = StableDiffusionPipeline.from_pretrained(
        config['base_model'],
        torch_dtype=torch.float16
    )
    
    lora_dir = Path(config['output_dir'])
    available_styles = [d.name for d in lora_dir.iterdir() if d.is_dir()] if lora_dir.exists() else []
    
    return base_model, available_styles


def style_transfer(content_image, style_name, style_strength, config, models):
    """
    Perform style transfer.
    
    Args:
        content_image: PIL Image
        style_name: Name of the style
        style_strength: Strength of style (0.0 to 1.0)
        config: Configuration dict
        models: Tuple of (base_model, available_styles)
    """
    base_model, available_styles = models
    
    if style_name not in available_styles:
        return None, f"Style '{style_name}' not found. Available: {available_styles}"
    
    lora_path = Path(config['output_dir']) / style_name / "final"
    
    if not lora_path.exists():
        return None, f"LoRA checkpoint not found at {lora_path}"
    
    base_model.unet.load_attn_procs(str(lora_path))
    
    content_image = content_image.resize((config['image_size'], config['image_size']))
    
    output_image = base_model(
        prompt="",
        image=content_image,
        num_inference_steps=50,
        guidance_scale=7.5,
        strength=style_strength
    ).images[0]
    
    return output_image, "Success"


def create_demo(config_path: str = "src/configs/lora_config.yaml"):
    """Create Gradio demo interface."""
    config = load_config(config_path)
    models = load_models(config)
    base_model, available_styles = models
    
    def process(content_image, style_name, style_strength):
        if content_image is None:
            return None, "Please upload a content image"
        
        output_image, message = style_transfer(
            content_image, style_name, style_strength, config, models
        )
        return output_image, message
    
    with gr.Blocks(title="Style Transfer Demo") as demo:
        gr.Markdown("# Customized Image Generation - Style Transfer")
        gr.Markdown("Upload a content image and select a style to generate a styled image.")
        
        with gr.Row():
            with gr.Column():
                content_input = gr.Image(label="Content Image", type="pil")
                style_dropdown = gr.Dropdown(
                    choices=available_styles,
                    label="Style",
                    value=available_styles[0] if available_styles else None
                )
                style_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                    label="Style Strength"
                )
                generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Output Image")
                status_text = gr.Textbox(label="Status")
        
        generate_btn.click(
            fn=process,
            inputs=[content_input, style_dropdown, style_strength],
            outputs=[output_image, status_text]
        )
        
        gr.Examples(
            examples=[],
            inputs=[content_input, style_dropdown, style_strength]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)

