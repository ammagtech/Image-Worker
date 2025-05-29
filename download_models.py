from huggingface_hub import snapshot_download
import os

# Define the models and LoRAs you need
models_to_download = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "latent-consistency/lcm-lora-sdxl",
    "nerijs/pixel-art-xl"
]

# Set the cache directory (should match HF_HOME in Dockerfile)
cache_dir = os.environ.get('HF_HOME', '/app/hf_cache')
os.makedirs(cache_dir, exist_ok=True)

print(f"Downloading models to cache directory: {cache_dir}")

for model_id in models_to_download:
    print(f"Downloading {model_id}...")
    snapshot_download(repo_id=model_id, cache_dir=cache_dir)
    print(f"Finished downloading {model_id}.")

print("All models and LoRAs downloaded successfully.")