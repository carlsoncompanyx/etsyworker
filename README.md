# ImageUpscaler

This repository provides a RunPod Serverless handler that upscales images with the
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) model
`RealESRGAN_x4plus`. The endpoint accepts one or more image URLs and produces 4× `.jpg`
outputs rendered with high-quality settings.

## Building the container

```bash
docker build -t image-upscaler .
```

The Docker image defaults to `runpod/pytorch:2.1.0-py3.10-cuda11.8`, installs the runtime
requirements for Real-ESRGAN, and prepares directories for cached model weights and
upscaled outputs. If your environment requires a different base image, pass
`--build-arg BASE_IMAGE=<image:tag>` to the `docker build` command. The handler downloads
the `RealESRGAN_x4plus` weights from Hugging Face on first use and reuses them across
invocations.

## Running locally

```bash
docker run --rm -p 8080:8080 \
  image-upscaler
```

The RunPod serverless runtime will automatically start and serve the handler defined in
`handler.py`.

## Invoking the handler

Send a JSON payload with the image URLs to process. `prompt` and `prompts` inputs are
accepted for API compatibility, although Real-ESRGAN does not use textual guidance.

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

The handler downloads each image, upscales it with Real-ESRGAN, saves the results as
maximum-quality `.jpg` files, and returns a response structured as:

```json
{
  "outputs": [
    "/app/output/upscaled_<uuid>.jpg",
    "/app/output/upscaled_<uuid>.jpg"
  ]
}
```

Set the `REAL_ESRGAN_WEIGHTS_URL` environment variable if you need to override the default
weight location. Tiling parameters can be tuned with `REALESRGAN_TILE`,
`REALESRGAN_TILE_PAD`, and `REALESRGAN_PRE_PAD` environment variables when processing large
images on constrained GPUs.
