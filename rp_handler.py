import runpod
from diffusers import DiffusionPipeline, LCMScheduler
import torch
import base64
from io import BytesIO


# Load base model and LoRA weights
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    variant="fp16",
    torch_dtype=torch.float16
)

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Load LoRA adapters
pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
pipe.load_lora_weights("./pixel-art-xl.safetensors", adapter_name="pixel")

# Combine adapters with weights
pipe.set_adapters(["lora", "pixel"], adapter_weights=[1.0, 1.2])

# Move model to GPU
pipe.to("cuda")


def handler(event):
    print("Worker started...")
    input_data = event.get('input', {})
    print(f"Received input: {input_data}")

    prompt = input_data.get('prompt', "pixel, a cute corgi")

    negative_prompt = "3d render, realistic, blurry, low quality, bad anatomy"

    try:
        # Generate the image
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=28,
            guidance_scale=7.0
        )
        image = result.images[0]

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "prompt": prompt,
            "image_base64": image_base64
        }

    except Exception as e:
        print(f"Error during image generation: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
