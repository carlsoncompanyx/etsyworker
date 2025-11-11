import io
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests
import torch
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_DIR", "/app/output"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get("UPSCALE_MODEL_ID", "stabilityai/stable-diffusion-x4-upscaler")
DEFAULT_PROMPT = os.environ.get(
    "UPSCALE_DEFAULT_PROMPT",
    "detailed, high-resolution, finely textured photograph"
)

_PIPELINE: Optional[StableDiffusionUpscalePipeline] = None


class ImageDownloadError(RuntimeError):
    """Raised when an input image cannot be downloaded."""


def _iter_prompts(image_count: int, prompt: Optional[str], prompts: Optional[Iterable[str]]) -> Iterable[str]:
    if prompts:
        prompt_list = list(prompts)
        if len(prompt_list) != image_count:
            raise ValueError("Length of 'prompts' must match 'image_urls'.")
        return prompt_list
    resolved = prompt or DEFAULT_PROMPT
    return [resolved] * image_count


def _download_image(url: str, download_dir: Path) -> Path:
    response = requests.get(url, timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - logged for debugging
        raise ImageDownloadError(f"Failed to download image from {url}: {exc}") from exc

    parsed = urlparse(url)
    extension = Path(parsed.path).suffix or ".png"
    filename = download_dir / f"source_{uuid.uuid4().hex}{extension}"
    filename.write_bytes(response.content)
    return filename


def _resolve_dtype_candidates(device: str) -> List[torch.dtype]:
    candidates: List[torch.dtype] = []
    preferred_name = os.environ.get("TORCH_DTYPE", "bfloat16")
    preferred = getattr(torch, preferred_name, None)
    if isinstance(preferred, torch.dtype):
        candidates.append(preferred)
    if device == "cuda":
        for dtype in (torch.float16, torch.float32):
            if dtype not in candidates:
                candidates.append(dtype)
    else:
        if torch.float32 not in candidates:
            candidates.append(torch.float32)
    return candidates


def _initialize_pipeline() -> None:
    global _PIPELINE

    if _PIPELINE is not None:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device with xformers support is required for Stable Diffusion upscaling."
        )

    device = "cuda"
    token = os.environ.get("HUGGINGFACE_TOKEN")
    last_error: Optional[Exception] = None

    for dtype in _resolve_dtype_candidates(device):
        kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if token:
            kwargs["use_auth_token"] = token
        try:
            pipeline = StableDiffusionUpscalePipeline.from_pretrained(MODEL_ID, **kwargs)
            pipeline.set_progress_bar_config(disable=True)
            pipeline.to(device)
            pipeline.enable_attention_slicing()
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to enable xformers memory efficient attention; ensure xformers is installed"
                ) from exc
            _PIPELINE = pipeline
            return
        except Exception as exc:  # pragma: no cover - exercised in runtime environments
            last_error = exc
            continue

    raise RuntimeError("Failed to initialize diffusion pipeline") from last_error


def _load_image(path: Path) -> Image.Image:
    with path.open("rb") as file_obj:
        image = Image.open(io.BytesIO(file_obj.read()))
        return image.convert("RGB")


def _run_upscale(prompt: str, image_path: Path, output_path: Path) -> Path:
    if _PIPELINE is None:
        _initialize_pipeline()
    assert _PIPELINE is not None  # for type checkers

    low_res_image = _load_image(image_path)

    with torch.inference_mode():
        result = _PIPELINE(prompt=prompt, image=low_res_image)
    upscaled = result.images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    upscaled.save(output_path, format="WEBP", quality=100)
    return output_path


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    input_payload = event.get("input") or {}
    image_urls = input_payload.get("image_urls")
    if not image_urls or not isinstance(image_urls, list):
        raise ValueError("'image_urls' must be a non-empty list of URLs.")

    prompt = input_payload.get("prompt")
    prompts = input_payload.get("prompts")

    outputs: List[str] = []
    temp_dir = Path("/tmp") / f"hf_upscaler_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        prompt_iterable = _iter_prompts(len(image_urls), prompt, prompts)
        for url, prompt_text in zip(image_urls, prompt_iterable):
            source_path = _download_image(url, temp_dir)
            output_name = f"upscaled_{uuid.uuid4().hex}.webp"
            final_output_path = OUTPUT_ROOT / output_name
            upscaled_path = _run_upscale(prompt_text, source_path, final_output_path)
            outputs.append(str(upscaled_path))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"outputs": outputs}
