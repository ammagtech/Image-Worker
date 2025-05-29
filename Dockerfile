# Use a slim Python base image
FROM python:3.10-slim

# Install system dependencies (git, ffmpeg, libsndfile1 are good choices for ML/audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Update pip to the latest version for better dependency resolution
RUN pip install --no-cache-dir --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Install PyTorch with a specific version and CUDA compatibility
# This ensures you have the correct backend for GPU acceleration
RUN pip install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 \
    && python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)" > /app/pytorch_version.txt

# Install all Python dependencies directly
# This is where you would list runpod and all diffusers related libraries
RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.27.0 \
    transformers==4.38.2 \
    accelerate==0.27.2 \
    safetensors==0.4.2 \
    Pillow==10.2.0 \
    xformers==0.0.25 \


# Copy the rest of your application code into the container
COPY . .

# Set the command to run your application when the container starts
CMD ["python", "rp_handler.py"]