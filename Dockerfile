FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /app

# Install PyTorch (CUDA 12.6)
RUN pip install --no-cache-dir torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126

# Copy project files
COPY . .

# Install dependencies before downloading models
RUN pip install --no-cache-dir \
    diffusers \
    accelerate \
    safetensors \
    transformers \
    pillow \
    numpy \
    tokenizers \
    peft==0.9.0 \
    runpod

# Download models into cache
RUN python3 download_models.py

# Run handler
CMD ["python3", "rp_handler.py"]
