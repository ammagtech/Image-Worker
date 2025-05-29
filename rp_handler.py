import runpod
from diffusers import DiffusionPipeline, LCMScheduler
import torch
import base64
from io import BytesIO
import traceback


print("Initializing model pipeline...")

try:
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

    # Load the base model
    print(f"Loading base model: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        variant="fp16",
        torch_dtype=torch.float16
    )

    print("Base model loaded. Setting scheduler...")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Load LoRA weights
    print(f"Loading LCM LoRA: {lcm_lora_id}")
    pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")

    print("Loading Pixel Art LoRA from ./pixel-art-xl.safetensors...")
    pipe.load_lora_weights("./pixel-art-xl.safetensors", adapter_name="pixel")

    # Set adapter weights
    print("Setting adapters and weights...")
    pipe.set_adapters(["lora", "pixel"], adapter_weights=[1.0, 1.2])

    # Move to CUDA
    print("Moving pipeline to CUDA...")
    pipe.to("cuda")

    print("Model pipeline successfully initialized.")
except Exception as e:
    print("🔥 Model initialization failed:")
    traceback.print_exc()
    raise e


def handler(event):
    print("🟢 Worker started...")
    try:
        input_data = event.get('input', {})
        print(f"📥 Received input: {input_data}")

        prompt = input_data.get('prompt', "pixel, a cute corgi")
        print(f"🧠 Prompt: {prompt}")

        negative_prompt = "3d render, realistic, blurry, low quality, bad anatomy"

        print("🎨 Generating image...")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=28,
            guidance_scale=7.0
        )
        image = result.images[0]

        print("🖼️ Image generated. Encoding to base64...")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("✅ Image encoding complete. Returning response.")
        return {
            "status": "success",
            "prompt": prompt,
            "image_base64": image_base64
        }

    except Exception as e:
        print("❌ Error during handler execution:")
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    print("🚀 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
