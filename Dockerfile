# Base image with CUDA support
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface/transformers
ENV HF_DATASETS_CACHE=/runpod-volume/huggingface/datasets

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler script and test
COPY handler.py .
COPY test_imports.py .

# Test that all imports work
RUN python test_imports.py

# Note: Models will be downloaded on first run and cached to /runpod-volume
# This is faster and more reliable than downloading during Docker build

# Start the unified handler
CMD ["python", "-u", "handler.py"]
