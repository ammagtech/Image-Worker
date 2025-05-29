import runpod
from diffusers import DiffusionPipeline, LCMScheduler
import torch
import io
import base64

# Global model loading (runs once per container start)
print("Initializing container...")

device = "cuda"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
pixel_lora_id = "nerijs/pixel-art-xl"
lora_weight = 1.0
pixel_weight = 1.2

# Load pipeline once
print("Loading diffusion pipeline...")
pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

print("Loading LoRA weights...")
pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
pipe.load_lora_weights(pixel_lora_id, adapter_name="pixel")
pipe.set_adapters(["lora", "pixel"], adapter_weights=[lora_weight, pixel_weight])

print("Moving pipeline to device...")
pipe.to(device=device, dtype=torch.float16)

print("Pipeline ready.")

# Handler called for each request
def handler(event):
    print("Worker Start")
    input_data = event.get('input', {})
    print(f"Received input: {input_data}")

    prompt = input_data.get('prompt', "pixel, a cute corgi")
    negative_prompt = "3d render, realistic, blurry, low quality, bad anatomy"
    num_inference_steps = 8
    guidance_scale = 1.5

    print(f"Generating image with prompt: {prompt}")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    print("Image generated. Converting to base64...")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

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

# Start serverless handler
if __name__ == '__main__':
    print("Starting serverless handler...")
    runpod.serverless.start({'handler': handler})
