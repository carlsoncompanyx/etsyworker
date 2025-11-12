import io
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_DIR", "/app/output"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("REAL_ESRGAN_MODEL", "RealESRGAN_x4plus")
MODEL_SCALE = int(os.environ.get("REAL_ESRGAN_SCALE", "4"))
MODEL_WEIGHTS_URL = os.environ.get(
    "REAL_ESRGAN_WEIGHTS_URL",
    "https://huggingface.co/xinntao/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.pth",
)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_UPSAMPLER: Optional[RealESRGANer] = None


class ImageDownloadError(RuntimeError):
    """Raised when an input image cannot be downloaded."""


class ModelDownloadError(RuntimeError):
    """Raised when the Real-ESRGAN weights cannot be retrieved."""


def _validate_prompts(image_count: int, prompt: Optional[str], prompts: Optional[Iterable[str]]) -> None:
    del prompt  # textual prompts are ignored by Real-ESRGAN
    if not prompts:
        return
    prompt_list = list(prompts)
    if len(prompt_list) != image_count:
        raise ValueError("Length of 'prompts' must match 'image_urls'.")


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


def _download_weights() -> Path:
    parsed = urlparse(MODEL_WEIGHTS_URL)
    if not parsed.scheme:
        raise ModelDownloadError("MODEL_WEIGHTS_URL must be an absolute URL")

    destination = MODEL_DIR / Path(parsed.path).name
    if destination.exists():
        return destination

    response = requests.get(MODEL_WEIGHTS_URL, timeout=300)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise ModelDownloadError(
            f"Failed to download Real-ESRGAN weights from {MODEL_WEIGHTS_URL}: {exc}"
        ) from exc

    tmp_path = destination.with_suffix(".tmp")
    tmp_path.write_bytes(response.content)
    tmp_path.replace(destination)
    return destination


def _build_rrdb_model(scale: int) -> RRDBNet:
    if MODEL_NAME == "RealESRGAN_x4plus":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    raise RuntimeError(f"Unsupported Real-ESRGAN model: {MODEL_NAME}")


def _initialize_upsampler() -> None:
    global _UPSAMPLER

    if _UPSAMPLER is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = _download_weights()
    model = _build_rrdb_model(MODEL_SCALE)

    half_precision = device == "cuda"
    _UPSAMPLER = RealESRGANer(
        scale=MODEL_SCALE,
        model_path=str(weights_path),
        model=model,
        tile=int(os.environ.get("REALESRGAN_TILE", "0")),
        tile_pad=int(os.environ.get("REALESRGAN_TILE_PAD", "10")),
        pre_pad=int(os.environ.get("REALESRGAN_PRE_PAD", "0")),
        half=half_precision,
        device=torch.device(device),
    )


def _load_image(path: Path) -> Image.Image:
    with path.open("rb") as file_obj:
        image = Image.open(io.BytesIO(file_obj.read()))
        return image.convert("RGB")


def _run_upscale(image_path: Path, output_path: Path) -> Path:
    if _UPSAMPLER is None:
        _initialize_upsampler()
    assert _UPSAMPLER is not None  # for type checkers

    low_res_image = _load_image(image_path)
    image_array = np.array(low_res_image)

    with torch.inference_mode():
        upscaled_array, _ = _UPSAMPLER.enhance(image_array, outscale=MODEL_SCALE)

    upscaled = Image.fromarray(upscaled_array)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    upscaled.save(
        output_path,
        format="JPEG",
        quality=100,
        subsampling=0,
    )
    return output_path


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    input_payload = event.get("input") or {}
    image_urls = input_payload.get("image_urls")
    if not image_urls or not isinstance(image_urls, list):
        raise ValueError("'image_urls' must be a non-empty list of URLs.")

    prompt = input_payload.get("prompt")
    prompts = input_payload.get("prompts")
    _validate_prompts(len(image_urls), prompt, prompts)

    outputs: List[str] = []
    temp_dir = Path("/tmp") / f"realesrgan_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for url in image_urls:
            source_path = _download_image(url, temp_dir)
            output_name = f"upscaled_{uuid.uuid4().hex}.jpg"
            final_output_path = OUTPUT_ROOT / output_name
            upscaled_path = _run_upscale(source_path, final_output_path)
            outputs.append(str(upscaled_path))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"outputs": outputs}
