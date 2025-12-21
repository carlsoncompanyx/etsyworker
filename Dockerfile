# Base image with CUDA support
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

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

# Copy unified handler script
COPY handler.py .

# Pre-download models to speed up cold starts
RUN python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('playgroundai/playground-v2.5-1024px-aesthetic', torch_dtype='float16', variant='fp16')"
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-large-patch14'); CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')"

# Start the unified handler
CMD ["python", "-u", "handler.py"]
