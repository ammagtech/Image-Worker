# RunPod GPU Deployment Guide

This guide explains how to deploy your Bark AI model on RunPod with GPU support.

## Changes Made for GPU Support

### 1. Code Optimizations (`rp_handler.py`)
- **Model Loading**: Moved model initialization outside the handler to load once at startup
- **GPU Memory Management**: Added proper CUDA cache clearing and garbage collection
- **Mixed Precision**: Enabled automatic mixed precision (AMP) for faster inference
- **Error Handling**: Added comprehensive error handling for GPU operations
- **Logging**: Added detailed logging for debugging and monitoring
- **Model Warming**: Pre-warm the model for faster first inference

### 2. Docker Improvements (`Dockerfile`)
- **Updated Base Image**: Using PyTorch 2.1.0 with CUDA 12.1
- **CUDA Environment**: Set proper NVIDIA environment variables
- **Dependencies**: Added build tools and updated package versions
- **CUDA Verification**: Added CUDA availability check during build

### 3. Dependencies (`requirements.txt`)
- **Updated PyTorch**: Version 2.1.0 for better GPU support
- **Updated Transformers**: Version 4.35.2 for latest features
- **Added Accelerate**: For better GPU memory management
- **Pinned Versions**: Specific versions for reproducibility

## Deployment Steps

### 1. Build and Test Locally (Optional)
```bash
# Build the Docker image
docker build -t bark-gpu .

# Test GPU functionality
docker run --gpus all bark-gpu python test_gpu.py

# Test the handler
docker run --gpus all -p 8000:8000 bark-gpu
```

### 2. Deploy on RunPod

#### Option A: Using RunPod CLI
```bash
# Install RunPod CLI
pip install runpod

# Login to RunPod
runpod login

# Deploy the serverless function
runpod deploy --name "bark-gpu" --gpu-type "RTX4090" .
```

#### Option B: Using RunPod Web Interface
1. Go to [RunPod Serverless](https://www.runpod.io/serverless)
2. Click "New Endpoint"
3. Choose "Custom Image" 
4. Upload your Docker image or connect to your GitHub repo
5. Select GPU type (recommended: RTX 4090, A100, or RTX 3090)
6. Set environment variables if needed
7. Deploy

### 3. Configuration Settings

#### Recommended GPU Types
- **RTX 4090**: Best price/performance for Bark model
- **RTX 3090**: Good alternative with 24GB VRAM
- **A100**: Best for production workloads
- **RTX 3080**: Minimum recommended (10GB VRAM)

#### Memory Requirements
- **Minimum**: 8GB VRAM
- **Recommended**: 16GB+ VRAM
- **Optimal**: 24GB+ VRAM

#### Environment Variables
```
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
CUDA_VISIBLE_DEVICES=0
```

## Testing Your Deployment

### 1. Test GPU Functionality
Use the included test script:
```bash
python test_gpu.py
```

### 2. Test API Endpoint
```bash
curl -X POST https://your-endpoint-url/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Hello, this is a test of the Bark model on GPU!",
      "voicePreset": "v2/en_speaker_6"
    }
  }'
```

### 3. Test with Python
```python
import requests
import json

url = "https://your-endpoint-url/runsync"
payload = {
    "input": {
        "prompt": "Hello world, testing GPU acceleration!",
        "voicePreset": "v2/en_speaker_6"
    }
}

response = requests.post(url, json=payload)
result = response.json()

if result["status"] == "success":
    print("✅ GPU deployment working!")
    # Save the audio
    import base64
    audio_data = base64.b64decode(result["audio_base64"])
    with open("test_output.wav", "wb") as f:
        f.write(audio_data)
else:
    print(f"❌ Error: {result.get('message', 'Unknown error')}")
```

## Performance Expectations

### GPU vs CPU Performance
- **CPU (16 cores)**: ~30-60 seconds per request
- **RTX 3080**: ~3-8 seconds per request
- **RTX 4090**: ~2-5 seconds per request
- **A100**: ~1-3 seconds per request

### Memory Usage
- **Model Loading**: ~3-4GB VRAM
- **Inference**: +1-2GB VRAM per request
- **Peak Usage**: ~6-8GB VRAM

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use smaller model
   - Enable gradient checkpointing
   - Use CPU offloading

2. **Model Loading Timeout**
   - Increase container startup timeout
   - Use model caching
   - Pre-download models

3. **Slow First Request**
   - Model warming is included in initialization
   - Consider keeping one instance warm

### Debug Commands
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Check model loading
python test_gpu.py
```

## Cost Optimization

1. **Use Spot Instances**: 50-80% cost savings
2. **Auto-scaling**: Scale down during low usage
3. **Model Caching**: Reduce cold start times
4. **Batch Processing**: Process multiple requests together

## Support

If you encounter issues:
1. Check the logs for detailed error messages
2. Verify GPU availability with `test_gpu.py`
3. Ensure your RunPod account has GPU credits
4. Contact RunPod support for infrastructure issues
