import runpod
from diffusers import DiffusionPipeline, LCMScheduler
import torch
import base64
from io import BytesIO
import os

device = "cuda"

def load_pipeline():
    print("Loading model and pipeline...")

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
    pixel_lora_path = "./pixel-art-xl.safetensors"

    print(f"Base model: {model_id}")
    print(f"LCM LoRA: {lcm_lora_id}")
    print(f"Pixel art LoRA path: {pixel_lora_path}")

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        variant="fp16",
        torch_dtype=torch.float32
    )


    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    print("Loading LoRA adapters...")
    pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")

    if not os.path.exists(pixel_lora_path):
        raise FileNotFoundError(f"{pixel_lora_path} not found!")

    pipe.load_lora_weights(pixel_lora_path, adapter_name="pixel")
    pipe.set_adapters(["lora", "pixel"], adapter_weights=[1.0, 1.2])

    pipe.to(device)

    print("Pipeline loaded and ready.")
    return pipe


# Global so model doesn't reload on every request
pipe = load_pipeline()

def handler(event):
    print("Worker started...")
    try:
        input_data = event.get('input', {})
        print(f"Input received: {input_data}")

        prompt = input_data.get('prompt', "pixel, a cute corgi")
        print(f"Prompt: {prompt}")

        negative_prompt = "3d render, realistic, blurry, low quality, bad anatomy"

        print("Starting image generation...")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=28,
            guidance_scale=7.0
        )

        image = result.images[0]
        print("Image generated.")

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("Image converted to base64.")

        return {
            "status": "success",
            "prompt": prompt,
            "image_base64": image_base64,
            "format": "png",
            "device": device
        }

    except Exception as e:
        print(f"Exception occurred: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
