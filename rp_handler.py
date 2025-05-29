import runpod
from diffusers import DiffusionPipeline, LCMScheduler
import torch
import base64
from io import BytesIO

device = "cuda"

def load_pipeline():
    print("Loading model and pipeline...")

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

    print(f"Base model: {model_id}")
    print(f"LCM LoRA: {lcm_lora_id}")

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        variant="fp16",
        torch_dtype=torch.float16  # Changed from float32 to float16 for better GPU performance
    )

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    print("Loading LCM LoRA adapter...")
    pipe.load_lora_weights(lcm_lora_id)  # Simplified - no need for adapter_name when using single LoRA

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

        prompt = input_data.get('prompt', "a cute corgi")  # Removed "pixel" from default prompt
        print(f"Prompt: {prompt}")

        negative_prompt = input_data.get('negative_prompt', "3d render, realistic, blurry, low quality, bad anatomy")

        # Get parameters from input with defaults optimized for LCM
        num_inference_steps = input_data.get('num_inference_steps', 8)  # LCM works well with 4-8 steps
        guidance_scale = input_data.get('guidance_scale', 1.5)  # Lower guidance for LCM

        print("Starting image generation...")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
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
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
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