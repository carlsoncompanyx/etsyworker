ARG BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
FROM ${BASE_IMAGE}

ENV OUTPUT_DIR=/app/output \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface

RUN mkdir -p "$OUTPUT_DIR" "$HUGGINGFACE_HUB_CACHE"

WORKDIR /app

COPY requirements.txt ./
RUN pip uninstall -y numpy || true && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade numpy==1.26.4 && \
    sed -i 's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|' /usr/local/lib/python3.12/dist-packages/basicsr/data/degradations.py


COPY . .

CMD ["python", "server.py"]
