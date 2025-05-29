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

# Install core Python dependencies needed for Hugging Face download
RUN pip install --no-cache-dir \
    huggingface_hub \
    diffusers \
    transformers \
    safetensors \
    accelerate # Added accelerate as it's a common dependency for diffusers

# --- ADDED: Pre-download models and LoRAs ---
# Set environment variable to store Hugging Face cache in a known location
ENV HF_HOME=/app/hf_cache

# Create the cache directory
RUN mkdir -p ${HF_HOME}

# Use a Python script to download models and LoRAs to ensure they are cached
COPY download_models.py .
RUN python download_models.py

# Remove the download script if not needed after build
RUN rm download_models.py
# --- END ADDED ---

# Copy project code
COPY . .

# Install remaining Python dependencies (might be redundant if already installed above, but good for clarity)
RUN pip install --no-cache-dir \
    runpod \
    pillow \
    numpy \
    tokenizers

# Start your app
CMD ["python3", "rp_handler.py"]