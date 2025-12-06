# Dockerfile for RunPod Serverless GPU Endpoint
# Base image with CUDA support
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download Real-ESRGAN models during build (saves cold start time)
RUN mkdir -p /app/models && \
    cd /app/models && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_anime_6B.pth && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth

# Copy your worker script
COPY handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Create cache directory
RUN mkdir -p /app/cache

# Expose port (not required for RunPod but OK)
EXPOSE 8000

# Run the worker
CMD ["python", "-u", "handler.py"]
