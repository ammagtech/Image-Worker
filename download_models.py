from diffusers import DiffusionPipeline
import torch

print("Downloading base model...")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16")
pipe.to("cpu")  # Use CPU to avoid GPU requirement during download

print("Downloading LoRA adapters...")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lora")
pipe.load_lora_weights("nerijs/pixel-art-xl", adapter_name="pixel")

print("Models downloaded.")
