# Use a slim Python base image for smaller image size
FROM python:3.10-slim

# Install system dependencies
# `git` is needed for cloning repositories if you use them,
# `ffmpeg` and `libsndfile1` are often needed for audio/video processing,
# which can be indirect dependencies for some ML applications.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache to keep image small

# Update pip to the latest version for better dependency resolution and security
RUN pip install --no-cache-dir --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Install PyTorch with a specific version and CUDA compatibility
# This is crucial for GPU acceleration.
RUN pip install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 \
    && python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)" > /app/pytorch_version.txt

# --- IMPORTANT: Pre-download Hugging Face Models During Build ---
# This step bakes the model files directly into your Docker image.
# It makes the image much larger but eliminates runtime downloads on cold starts.
# We set HF_HOME to a specific directory within the container for clarity,
# and ensure it's created.

ENV HF_HOME=/usr/local/huggingface_cache
RUN mkdir -p ${HF_HOME}

# Download the base Stable Diffusion XL model
RUN python -c "import os; os.environ['HF_HOME'] = '${HF_HOME}'; from diffusers import DiffusionPipeline; print('Downloading stabilityai/stable-diffusion-xl-base-1.0...'); DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', variant='fp16', torch_dtype=torch.float16)"

# Download the LCM LoRA model
RUN python -c "import os; os.environ['HF_HOME'] = '${HF_HOME}'; from huggingface_hub import hf_hub_download; print('Downloading latent-consistency/lcm-lora-sdxl...'); hf_hub_download(repo_id='latent-consistency/lcm-lora-sdxl', filename='pytorch_lora_weights.safetensors')"

# Download the Pixel Art LoRA model
RUN python -c "import os; os.environ['HF_HOME'] = '${HF_HOME}'; from huggingface_hub import hf_hub_download; print('Downloading nerijs/pixel-art-xl...'); hf_hub_download(repo_id='nerijs/pixel-art-xl', filename='pytorch_lora_weights.safetensors')"
# --- END Pre-download Section ---


# Copy your application code into the container
# This comes AFTER model downloads to ensure models are cached before your app uses them.
# It also helps with Docker caching: if only your code changes, this layer and subsequent
# ones are rebuilt, but not the heavy dependency/model download layers.
COPY . .

# Install Python dependencies
# Ensure all libraries your rp_handler.py needs are listed here.
# Pinning versions (e.g., diffusers==0.27.0) is highly recommended for stability.
RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.27.0 \
    accelerate==0.27.2 \
    pillow==10.2.0 \
    numpy==1.26.4 \
    tokenizers==0.19.1 \
    safetensors==0.4.2 \
    transformers==4.39.3 \
    peft==0.9.0 \
    # You might need to add `scipy` if diffusers depends on it for certain features
    # scipy # Uncomment if you get a dependency error for scipy

# Start your application
# Use `python` instead of `python3` as `python` is usually symlinked to `python3`
# in Python Docker images, making it slightly more robust.
CMD ["python", "rp_handler.py"]