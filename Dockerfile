FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --no-cache-dir --upgrade pip

# Working directory
WORKDIR /app

# Install PyTorch 2.7.0 with CUDA 12.6
RUN pip install --no-cache-dir torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126 \
    && python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)" > /app/pytorch_version.txt
# Copy project code
COPY . .

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
    peft==0.9.0 
# Start your app
CMD ["python3", "rp_handler.py"]