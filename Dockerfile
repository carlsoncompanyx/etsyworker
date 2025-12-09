# Dockerfile for RunPod Serverless GPU Endpoint
# Using official PyTorch base image (always available)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Create models directory and download Real-ESRGAN models during build
RUN mkdir -p /app/models && \
    cd /app/models && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_anime_6B.pth && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth && \
    echo "✅ All Real-ESRGAN models downloaded"

# Copy the worker script
COPY runpod_worker.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache

# Create cache directory
RUN mkdir -p /app/cache

# Expose port (not strictly necessary for RunPod but good practice)
EXPOSE 8000

# Run the worker
CMD ["python", "-u", "runpod_worker.py"]
