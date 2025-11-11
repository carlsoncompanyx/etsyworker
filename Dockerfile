ARG BASE_IMAGE=runpod/pytorch:2.1.0-py3.10-cuda11.8
FROM ${BASE_IMAGE}

ENV OUTPUT_DIR=/app/output \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface

RUN mkdir -p "$OUTPUT_DIR" "$HUGGINGFACE_HUB_CACHE"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "server.py"]
