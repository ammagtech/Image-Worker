# Use a slim Python base image for smaller image size
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Update pip to the latest version for better dependency resolution and security
RUN pip install --no-cache-dir --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Install PyTorch with a specific version and CUDA compatibility
RUN pip install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 \
    && python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)" > /app/pytorch_version.txt

# Install all Python dependencies including PEFT
# Pinning versions is highly recommended for stability.
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
    scipy # Add scipy as it's often a dependency for diffusers/transformers

# --- IMPORTANT: Pre-download Hugging Face Models During Build ---
# This step bakes the model files directly into your Docker image.
# It makes the image much larger but eliminates runtime downloads on cold starts.
# We set HF_HOME to a specific directory within the container for clarity.

ENV HF_HOME=/usr/local/huggingface_cache
RUN mkdir -p ${HF_HOME}

# Download the base Stable Diffusion XL model
# Setting HF_HOME in os.environ for the python command ensures it's used.
# The from_pretrained method will handle the caching structure within HF_HOME.
RUN python -c "import os; os.environ['HF_HOME'] = '${HF_HOME}'; from diffusers import DiffusionPipeline; print('Downloading stabilityai/stable-diffusion-xl-base-1.0...'); DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', variant='fp16', torch_dtype=torch.float16, low_cpu_mem_usage=True)"

# Download the LoRA models using hf_hub_download
# The `cache_dir` argument for hf_hub_download is often the *root* of the HF cache (HF_HOME),
# and hf_hub_download will create the `datasets--repo-owner--repo-name` structure itself.
RUN python -c "import os; os.environ['HF_HOME'] = '${HF_HOME}'; from huggingface_hub import hf_hub_download; print('Downloading latent-consistency/lcm-lora-sdxl...'); hf_hub_download(repo_id='latent-consistency/lcm-lora-sdxl', filename='pytorch_lora_weights.safetensors', cache_dir=os.environ['HF_HOME'])"
RUN python -c "import os; os.environ['HF_HOME'] = '${HF_HOME}'; from huggingface_hub import hf_hub_download; print('Downloading nerijs/pixel-art-xl...'); hf_hub_download(repo_id='nerijs/pixel-art-xl', filename='pytorch_lora_weights.safetensors', cache_dir=os.environ['HF_HOME'])"

# --- END Pre-download Section ---


# Copy your application code into the container
COPY . .

# Start your application
CMD ["python", "rp_handler.py"]