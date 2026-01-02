# Use a highly compatible CUDA runtime
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies for OpenCV and Image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Optimize Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy handler code
COPY handler.py .

# Add a simple health check that prints versions
RUN python -c "import sys; print(f'Python: {sys.version}'); import torch; print(f'PyTorch: {torch.__version__}'); import diffusers; print(f'Diffusers: {diffusers.__version__}')"

CMD ["python", "-u", "handler.py"]
