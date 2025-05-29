# Use a Python 3.10 slim image
FROM python:3.10-slim

# Set environment variable to avoid Python buffering issues
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Optional sanity check (logs to file inside container)
RUN python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" > /app/pytorch_version.txt

# Copy project files
COPY . .

# Download the pixel art LoRA weights from Civitai
RUN curl -L -o pixel-art-xl.safetensors https://civitai.com/api/download/models/140134

# Install Python dependencies
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

# Command to run your handler
CMD ["python3", "-u", "rp_handler.py"]
