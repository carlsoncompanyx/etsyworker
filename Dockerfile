FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# ---------- Environment setup ----------
ENV OUTPUT_DIR=/app/output \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ---------- Install dependencies ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
        wget \
        nodejs \
        npm \
        libgl1 \
        libglib2.0-0 \
        libgtk-3-0 \
        libnss3 \
        libasound2 \
        libatk-bridge2.0-0 \
        libx11-xcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libxkbcommon0 \
        libpango-1.0-0 \
        libgbm1 \
        libatk1.0-0 \
        xz-utils \
        tar && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# The CRITICAL FIX line has been removed from here!
# DNS will be configured via the docker build --dns flag.

# ---------- Build Upscayl CLI from source (Should now succeed) ----------
RUN git clone https://github.com/upscayl/upscayl-cli.git /opt/upscayl && \
    cd /opt/upscayl && npm install && npm run build && npm install -g .

# ---------- Python dependencies ----------
RUN pip install --no-cache-dir runpod requests

# ---------- Prepare output directory ----------
RUN mkdir -p /app/output

COPY handler.py ./

CMD ["python", "-u", "handler.py"]
