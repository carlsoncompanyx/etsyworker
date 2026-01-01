# handler.py
from __future__ import annotations

import base64
import io
import os
import sys
import traceback
from typing import Any, Dict, Optional


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".write_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
        return True
    except Exception:
        return False


def _resolve_cache_root() -> str:
    preferred = "/runpod-volume/huggingface"
    fallback = "/tmp/huggingface"

    if os.path.isdir("/runpod-volume") and _is_writable_dir(preferred):
        return preferred

    _is_writable_dir(fallback)
    return fallback


CACHE_ROOT = _resolve_cache_root()

# IMPORTANT: override (NOT setdefault) so Dockerfile/env can't pin us to a bad path
os.environ["HF_HOME"] = CACHE_ROOT
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_ROOT, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_ROOT, "hub")
os.environ["DIFFUSERS_CACHE"] = os.path.join(CACHE_ROOT, "diffusers")

print(f"[boot] python={sys.version.split()[0]} cache_root={CACHE_ROOT}", flush=True)

import runpod  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from diffusers import DiffusionPipeline, EDMDPMSolverMultistepScheduler  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402


PLAYGROUND_REPO = "playgroundai/playground-v2.5-1024px-aesthetic"
AESTHETICS_REPO = "discus0434/aesthetic-predictor-v2-5"

CFG = 7.0
CREATE_STEPS = 25
PRODUCTION_STEPS = 50

pipe: Optional[DiffusionPipeline] = None
aesthetic_model: Optional[torch.nn.Module] = None
aesthetic_processor: Optional[Any] = None


def _log(msg: str) -> None:
    print(msg, flush=True)


def _img_to_b64_jpeg(img: Image.Image, quality: int = 95) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_pipe() -> None:
    global pipe
    if pipe is not None:
        return

    _log("[init] loading diffusion pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        PLAYGROUND_REPO,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=CACHE_ROOT,
    ).to("cuda")
    pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    _log("[init] pipeline ready")


def load_aesthetics() -> None:
    global aesthetic_model, aesthetic_processor
    if aesthetic_model is not None and aesthetic_processor is not None:
        return

    _log("[init] loading aesthetic predictor...")
    aesthetic_model = AutoModel.from_pretrained(
        AESTHETICS_REPO,
        trust_remote_code=True,
        cache_dir=CACHE_ROOT,
    ).to("cuda")
    aesthetic_processor = AutoProcessor.from_pretrained(
        AESTHETICS_REPO,
        trust_remote_code=True,
        cache_dir=CACHE_ROOT,
    )
    _log("[init] aesthetic predictor ready")


def _calc_score(img: Image.Image) -> float:
    assert aesthetic_model is not None and aesthetic_processor is not None
    inputs = aesthetic_processor(images=img, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        outputs = aesthetic_model(**inputs)
    return round(float(outputs.logits.reshape(-1)[0].item()), 2)


def route_create(inp: Dict[str, Any]) -> Dict[str, Any]:
    load_pipe()
    load_aesthetics()
    assert pipe is not None

    prompt = str(inp.get("prompt", ""))
    negative = str(inp.get("negative_prompt", ""))
    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))

    seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    gen = torch.Generator(device="cuda").manual_seed(seed)

    _log(f"[create] {width}x{height} seed={seed} steps={CREATE_STEPS} cfg={CFG}")

    with torch.inference_mode():
        img = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=CREATE_STEPS,
            guidance_scale=CFG,
            generator=gen,
        ).images[0]

    score = _calc_score(img)

    return {
        "route": "create",
        "image": _img_to_b64_jpeg(img),
        "aesthetic_score": score,
        "metadata": {
            "seed": seed,
            "steps": CREATE_STEPS,
            "cfg_scale": CFG,
            "width": width,
            "height": height,
        },
    }


def route_production(inp: Dict[str, Any]) -> Dict[str, Any]:
    load_pipe()
    assert pipe is not None

    prompt = str(inp.get("prompt", ""))
    negative = str(inp.get("negative_prompt", ""))
    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))

    if "seed" not in inp:
        raise ValueError("production requires 'seed' from create")

    seed = int(inp["seed"])
    gen = torch.Generator(device="cuda").manual_seed(seed)

    _log(f"[production] {width}x{height} seed={seed} steps={PRODUCTION_STEPS} cfg={CFG}")

    with torch.inference_mode():
        img = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=PRODUCTION_STEPS,
            guidance_scale=CFG,
            generator=gen,
        ).images[0]

    return {
        "route": "production",
        "image": _img_to_b64_jpeg(img),
        "metadata": {
            "seed": seed,
            "steps": PRODUCTION_STEPS,
            "cfg_scale": CFG,
            "width": width,
            "height": height,
        },
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input")
        if inp == "health":
            return {
                "status": "healthy",
                "gpu_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "cache_root": CACHE_ROOT,
            }

        assert isinstance(inp, dict)
        route = str(inp.get("route", "")).lower()

        if route == "create":
            return route_create(inp)
        if route == "production":
            return route_production(inp)

        return {"error": "route must be 'create' or 'production'"}

    except Exception as e:
        _log(f"[ERR] {e}")
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    _log("[boot] starting runpod serverless loop...")
    runpod.serverless.start({"handler": handler})
