# Use a RunPod base image that includes PyTorch and CUDA 12.1
# This saves you from installing PyTorch manually and ensures GPU compatibility.
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install the remaining Python packages for your application
# PyTorch is already in the base image, so we don't need to install it again.
RUN pip install --no-cache-dir \
    runpod \
    diffusers \
    transformers \
    accelerate \
    Pillow

# Copy your handler file into the container
# Ensure your Python file is named 'rp_handler.py' to match this line
COPY rp_handler.py .

# Command to run the handler when the pod starts
CMD ["python3", "-u", "rp_handler.py"]