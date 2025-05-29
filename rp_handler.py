import runpod
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import base64
from io import BytesIO

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.load_lora_weights("nerijs/pixel-art-xl")
pipe.to("cuda")

def handler(event):
    print("Worker Start")
    input_data = event.get('input', {})
    print(f"Received input: {input_data}")

    # Only take prompt as input, keep other parameters static for best pixel art results
    prompt = input_data.get('prompt', "pixel, a cute corgi")

    # Static parameters optimized for pixel art generation
    negative_prompt = "3d render, realistic, blurry, low quality, bad anatomy"


    # Generate the image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=7.0
    ).images[0]

    # Convert the image to a base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image_base64": image_base64,
    }


if __name__ == '__main__':
    print("Starting serverless handler...")
    runpod.serverless.start({'handler': handler})
