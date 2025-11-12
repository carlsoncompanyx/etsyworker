FROM xinntao/realesrgan:latest

ENV PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/app/output

WORKDIR /app

# Add only what you actually need
COPY requirements.txt ./
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "server.py"]
