# Pixel Art AI Model - GPU Optimized RunPod Worker

This is a GPU-optimized serverless worker for pixel art image generation using Stable Diffusion XL with LoRA adapters. The worker loads models directly from Hugging Face and is designed to run efficiently on RunPod with CUDA GPU acceleration, providing fast pixel art image generation from text prompts.

## Features

- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Memory Efficient**: Proper GPU memory management and cleanup
- **Fast Inference**: Mixed precision and model warming for optimal performance
- **Error Handling**: Comprehensive error handling and logging
- **Scalable**: Designed for serverless deployment on RunPod

## Model Information

- **Base Model**: Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0)
- **LoRA Adapters**: LCM LoRA + Pixel Art LoRA (nerijs/pixel-art-xl)
- **Task**: Text-to-Image Generation (Pixel Art Style)
- **GPU Memory**: ~8-12GB VRAM required
- **Performance**: 3-10 seconds per request (depending on GPU and steps)

## Quick Start

### Local Testing

```bash
# 1. Create a Python virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test GPU functionality (optional)
python test_gpu.py

# 5. Run the handler locally
python rp_handler.py
```

### Docker Build and Test

```bash
# Build docker image with GPU support
docker build -t pixel-art-gpu:latest .

# Test with GPU (requires nvidia-docker)
docker run --gpus all pixel-art-gpu:latest python test_gpu.py

# Run the service
docker run --gpus all -p 8000:8000 pixel-art-gpu:latest
```

## RunPod Deployment

See [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for detailed deployment instructions.

### Quick Deploy
```bash
# Build and push to registry
docker build -t your-registry/pixel-art-gpu:v1.0.0 --platform linux/amd64 .
docker push your-registry/pixel-art-gpu:v1.0.0

# Deploy on RunPod using the web interface or CLI
```

## API Usage

### Request Format
```json
{
  "input": {
    "prompt": "pixel, a cute corgi",
    "negative_prompt": "3d render, realistic",
    "num_inference_steps": 8,
    "guidance_scale": 1.5,
    "lora_weight": 1.0,
    "pixel_weight": 1.2
  }
}
```

### Response Format
```json
{
  "status": "success",
  "prompt": "pixel, a cute corgi",
  "negative_prompt": "3d render, realistic",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "format": "png",
  "device": "cuda",
  "num_inference_steps": 8,
  "guidance_scale": 1.5
}
```

## Performance

| GPU Type | Inference Time | Memory Usage |
|----------|---------------|--------------|
| RTX 4090 | 3-6 seconds   | ~10GB VRAM  |
| RTX 3090 | 4-8 seconds   | ~12GB VRAM  |
| RTX 3080 | 6-12 seconds  | ~10GB VRAM  |
| CPU      | 60-180 seconds| ~8GB RAM    |

## Troubleshooting

1. **GPU not detected**: Check CUDA installation and Docker GPU support
2. **Out of memory**: Reduce batch size or use smaller GPU
3. **Slow performance**: Ensure GPU is being used and model is warmed up

Run `python test_gpu.py` to diagnose GPU issues.
