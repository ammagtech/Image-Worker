# Use a Python 3.10 slim image
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --no-cache-dir --upgrade pip

# Working directory inside the container
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121 \
    && python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" > /app/pytorch_version.txt

# Copy project code and the handler script
COPY . .

# Download the pixel art LoRA file from Civitai
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

# Start your app
CMD ["python3", "-u", "rp_handler.py"]
