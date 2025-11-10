FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV UPSCAYL_VERSION=latest \
    UPSCAYL_CLI_URL=https://github.com/upscayl/upscayl/releases/latest/download/upscayl-cli-linux.tar.xz \
    OUTPUT_DIR=/app/output

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tar \
        xz-utils \
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
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/upscayl /app/output && \
    curl -L "$UPSCAYL_CLI_URL" -o /tmp/upscayl-cli-linux.tar.xz && \
    tar -xJf /tmp/upscayl-cli-linux.tar.xz -C /opt/upscayl && \
    chmod +x /opt/upscayl/upscayl-cli && \
    ln -s /opt/upscayl/upscayl-cli /usr/local/bin/upscayl-cli && \
    rm /tmp/upscayl-cli-linux.tar.xz

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "server.py"]
