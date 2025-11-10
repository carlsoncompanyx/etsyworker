# ImageUpscaler

This repository provides a RunPod Serverless handler that uses the Upscayl CLI to upscale
images at 4× resolution with the `ultrasharp` model. The serverless endpoint accepts a list
of image URLs and returns the filesystem paths to the resulting `.webp` images within the
container.

## Building the container

```bash
docker build -t image-upscaler .
# Pin a specific release tag when needed
docker build -t image-upscaler --build-arg UPSCAYL_VERSION=v3.0.0 .
# Supply a GitHub token if you routinely hit API rate limits
docker build -t image-upscaler \
  --build-arg UPSCAYL_VERSION=latest \
  --build-arg GITHUB_TOKEN="$GITHUB_TOKEN" .
```

The Docker image extends `runpod/pytorch:2.3.0-py3.10-cuda12.1`, discovers the correct
Upscayl CLI release asset for Linux (defaulting to the `latest` tag), and prepares the
Python runtime. The build script supports `.tar.xz`, `.tar.gz`, and `.zip` release
archives and will automatically link the bundled `upscayl-cli` binary into the system
`PATH`.

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
