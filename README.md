# ImageUpscaler

This repository provides a RunPod Serverless handler that upscales images with the
[`stabilityai/stable-diffusion-x4-upscaler`](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)
DiffusionPipeline. The endpoint accepts one or more image URLs, optionally guided by a
prompt, and produces 4× `.webp` outputs rendered with high-quality settings.

## Building the container

```bash
docker build -t image-upscaler .
```

The Docker image defaults to `runpod/pytorch:2.1.0-py3.10-cuda11.8`, installs the runtime
dependencies for Diffusers, and prepares a cache directory at `/app/.cache/huggingface`
for downloaded model weights. If your environment requires a different base image, pass
`--build-arg BASE_IMAGE=<image:tag>` to the `docker build` command. Provide a
`HUGGINGFACE_TOKEN` build or runtime environment variable if you need to access gated
models or avoid anonymous rate limits.

## Running locally

```bash
docker run --rm -p 8080:8080 \
  -e HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  image-upscaler
```

The RunPod serverless runtime will automatically start and serve the handler defined in
`handler.py`.

## Invoking the handler

Send a JSON payload with the image URLs to process. You can optionally supply a single
`prompt` applied to every image, or a `prompts` list that matches the order of
`image_urls`.

```json
{
  "input": {
    "image_urls": [
      "https://example.com/image-1.png",
      "https://example.com/photo.jpg"
    ],
    "prompt": "Finely detailed, cinematic lighting"
  }
}
```

The handler downloads each image, upscales it with the Stable Diffusion x4 pipeline, saves
the results as maximum-quality `.webp` files, and returns a response structured as:

```json
{
  "outputs": [
    "/app/output/upscaled_<uuid>.webp",
    "/app/output/upscaled_<uuid>.webp"
  ]
}
```

Set the `UPSCALE_DEFAULT_PROMPT` environment variable to control the default guidance when
no prompt is provided. Model weights are cached inside the container and reused across
invocations for faster warm starts.
