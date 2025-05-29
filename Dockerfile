# Use a Python 3.10 slim image
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --no-cache-dir --upgrade pip

# Working directory inside the container
WORKDIR /app

# Install PyTorch with CUDA support
# Using cu121 (CUDA 12.1) is generally more stable and widely supported on RunPod instances
# For newer GPUs or if you specifically need the absolute latest, you might try cu124 or cu125/cu126 if available and verified by PyTorch.
# As of current, cu121 is a very safe bet.
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121 \
    && python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" > /app/pytorch_version.txt

# Copy project code and the handler script
COPY . .

# Install Python dependencies
# 'xformers' is highly recommended for Stable Diffusion models for memory efficiency and speed.
# 'safetensors' is a dependency for many Hugging Face models.
RUN pip install --no-cache-dir \
    runpod \
    diffusers \
    accelerate \
    pillow \
    numpy \
    tokenizers \
    safetensors \
    transformers \
    xformers

# Start your app (RunPod will call the handler function defined in rp_handler.py)
CMD ["python3", "-u", "rp_handler.py"]