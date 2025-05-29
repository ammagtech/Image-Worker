import runpod
from diffusers import DiffusionPipeline, LCMScheduler
import torch
import io
import base64


def handler(event):
    print("Worker Start")
    input_data = event.get('input', {})
    print(f"Received input: {input_data}")

    # Only take prompt as input, keep other parameters static for best pixel art results
    prompt = input_data.get('prompt', "pixel, a cute corgi")

    # Static parameters optimized for pixel art generation
    negative_prompt = "3d render, realistic, blurry, low quality, bad anatomy"
    num_inference_steps = 8
    guidance_scale = 1.5
    lora_weight = 1.0
    pixel_weight = 1.2

    print(f"Prompt: {prompt}")
    print(f"Using optimized static parameters for pixel art generation")

    device = "cuda"
    print(f"Using device: {device}")

    # Load the diffusion pipeline
    print("Loading diffusion pipeline...")
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
    pixel_lora_id = "nerijs/pixel-art-xl"

    pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    print("Pipeline loaded.")

    # Load LoRA weights directly from Hugging Face
    print("Loading LoRA weights...")
    pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
    pipe.load_lora_weights(pixel_lora_id, adapter_name="pixel")
    pipe.set_adapters(["lora", "pixel"], adapter_weights=[lora_weight, pixel_weight])
    print("LoRA weights loaded.")

    # Move to device
    print("Moving pipeline to device...")
    pipe.to(device=device, dtype=torch.float16)
    print("Pipeline moved to device.")

    # Generate image
    print("Generating image...")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    print("Image generated.")

    # Convert image to base64
    print("Converting image to base64...")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    print("Image conversion complete.")

    return {
        "status": "success",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image_base64": image_base64,
        "format": "png",
        "device": device,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale
    }


if __name__ == '__main__':
    print("Starting serverless handler...")
    runpod.serverless.start({'handler': handler})
