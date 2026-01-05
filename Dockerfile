# Use stable PyTorch 2.1.2 base
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /workspace /runpod-volume /network-volume /volume /data /mnt

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/huggingface/hub \
    TRANSFORMERS_CACHE=/workspace/huggingface/transformers \
    HF_HUB_DISABLE_TELEMETRY=1

COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
