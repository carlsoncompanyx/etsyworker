# ImageUpscaler

This repository provides a RunPod Serverless handler that uses the Upscayl CLI to upscale
images at 4× resolution with the `ultrasharp` model. The serverless endpoint accepts a list
of image URLs and returns the filesystem paths to the resulting `.webp` images within the
container.

## Building the container

```bash
docker build -t image-upscaler .
```

The Docker image extends `runpod/pytorch:2.3.0-py3.10-cuda12.1`, installs the latest Upscayl
CLI release, and prepares the Python runtime.

## Running locally

```bash
docker run --rm -p 8080:8080 image-upscaler
```

The RunPod serverless runtime will automatically start and serve the handler defined in
`handler.py`.

## Invoking the handler

Send a JSON payload with the image URLs to process:

```json
{
  "input": {
    "image_urls": [
      "https://example.com/image-1.png",
      "https://example.com/photo.jpg"
    ]
  }
}
```

The handler downloads each image, upscales it with Upscayl using the `ultrasharp` model at 4×
scale, saves the results as maximum-quality `.webp` files, and returns a response structured as:

```json
{
  "outputs": [
    "/app/output/upscaled_<uuid>.webp",
    "/app/output/upscaled_<uuid>.webp"
  ]
}
```

The `/app/output` directory is persisted within the container and can be mounted or exported as
needed when deploying the endpoint in RunPod.
