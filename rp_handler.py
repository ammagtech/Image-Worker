import runpod
from diffusers import DiffusionPipeline, LCMScheduler
import torch
import base64
from io import BytesIO

# Set the device globally. This will be 'cuda' if a GPU is available, otherwise 'cpu'.
# RunPod environments typically guarantee 'cuda'.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_pipeline():
    """
    Loads the Stable Diffusion XL base model and LCM LoRA.
    This function is called once when the worker starts.
    """
    print("Loading model and pipeline...")

    # Define the model and LoRA IDs
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

    print(f"Base model ID: {model_id}")
    print(f"LCM LoRA ID: {lcm_lora_id}")

    # Initialize the DiffusionPipeline
    # Use variant="fp16" and torch_dtype=torch.float16 for optimized performance on GPUs
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        variant="fp16",
        torch_dtype=torch.float16,
        # Optional: Add `trust_remote_code=True` if you encounter issues with custom models,
        # but generally not needed for official Hugging Face models.
        # trust_remote_code=True
    )

    # Configure the scheduler for LCM
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    print("Loading LCM LoRA adapter...")
    # Load the LoRA weights. No need for adapter_name when it's the only one being loaded.
    pipe.load_lora_weights(lcm_lora_id)

    # Move the pipeline to the specified device (GPU)
    pipe.to(device)

    print("Pipeline loaded and ready.")
    return pipe

# Load the pipeline globally so it's initialized only once when the worker container starts.
# This prevents reloading the model on every inference request.
pipe = load_pipeline()

def handler(event):
    """
    Main handler function for RunPod serverless requests.
    Processes the input event and generates an image.
    """
    print("Worker received request...")
    try:
        # Extract input data from the event. Default to an empty dictionary if 'input' is missing.
        input_data = event.get('input', {})
        print(f"Input data received: {input_data}")

        # Get prompt and negative prompt with sensible defaults
        prompt = input_data.get('prompt', "a cute corgi in a field of flowers, cinematic lighting")
        print(f"Prompt: {prompt}")

        negative_prompt = input_data.get('negative_prompt', "3d render, realistic, blurry, low quality, bad anatomy, deformed, watermark, text")
        print(f"Negative prompt: {negative_prompt}")


        # Get generation parameters with defaults optimized for LCM
        # LCM works well with a small number of inference steps (e.g., 4-8)
        num_inference_steps = input_data.get('num_inference_steps', 8)
        # Guidance scale for LCM is typically much lower (e.g., 1.0 - 2.0)
        guidance_scale = input_data.get('guidance_scale', 1.5)

        print(f"Generating image with {num_inference_steps} steps and guidance scale {guidance_scale}...")

        # Perform the image generation
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )

        image = result.images[0]
        print("Image generated successfully.")

        # Convert the PIL Image to a Base64 encoded PNG string
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("Image converted to base64.")

        # Return the results
        return {
            "status": "success",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "image_base64": image_base64,
            "format": "png", # Explicitly state the format
            "device": device # Confirm which device was used
        }

    except Exception as e:
        # Log the exception for debugging and return an error message
        print(f"An error occurred during request processing: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# This block ensures that runpod.serverless.start is called when the script is executed.
if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    # The handler dictionary maps event types to handler functions.
    # In this common setup, all requests are handled by the 'handler' function.
    runpod.serverless.start({"handler": handler})