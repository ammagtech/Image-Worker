# Use a base image with CUDA 12.1 support
FROM python:3.10-slim

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

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sanity check
RUN python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Available:', torch.cuda.is_available())" > /app/pytorch_version.txt

# Copy your project code
COPY . .

# Run the handler (main script for inference)
CMD ["python3", "-u", "rp_handler.py"]
