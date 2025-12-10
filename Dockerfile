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

# Downgrade NumPy first to avoid compatibility issues
RUN pip install --no-cache-dir "numpy<2.0.0,>=1.24.0"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Create models directory (models will be downloaded on first run)
RUN mkdir -p /app/models

# Copy the worker script
COPY handler.py .

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
CMD ["python", "-u", "handler.py"]
