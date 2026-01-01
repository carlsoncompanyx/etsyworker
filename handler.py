# /app/handler.py
from __future__ import annotations

import base64
import io
import os
import sys
import traceback
from typing import Any, Dict, Optional

# ---- cache path: use /runpod-volume if mounted+writable, else /tmp ----
def _pick_cache_root() -> str:
    preferred = "/runpod-volume"
    try:
        if os.path.isdir(preferred):
            test_dir = os.path.join(preferred, "huggingface")
            os.makedirs(test_dir, exist_ok=True)
            test_file = os.path.join(test_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            return preferred
    except Exception:
        pass

    fallback = "/tmp"
    os.makedirs(os.path.join(fallback, "huggingface"), exist_ok=True)
    return fallback


CACHE_ROOT = _pick_cache_root()
HF_HOME = os.path.join(CACHE_ROOT, "huggingface")

os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import runpod
import torch
from PIL import Image
from diffusers import DiffusionPipeline, EDMDPMSolverMultistepScheduler
from transformers import AutoModel, AutoProcessor

PLAYGROUND_REPO = "playgroundai/playground-v2.5-1024px-aesthetic"
AESTHETICS_REPO = "discus0434/aesthetic-predictor-v2-5"

CFG = 7.0
CREATE_STEPS = 25
PRODUCTION_STEPS = 50

pipe: Optional[DiffusionPipeline] = None
aesthetic_model: Optional[torch.nn.Module] = None
aesthetic_processor: Optional[Any] = None


def _log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def _img_to_b64_jpeg(img: Image.Image, quality: int = 95) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_pipe() -> None:
    global pipe
    if pipe is not None:
        return

    _log(f"[init] HF_HOME={os.environ.get('HF_HOME')}")
    _log("[init] Loading Playground v2.5 pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        PLAYGROUND_REPO,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    _log("[init] Playground ready.")


def load_aesthetics() -> None:
    global aesthetic_model, aesthetic_processor
    if aesthetic_model is not None and aesthetic_processor is not None:
        return

    _log("[init] Loading aesthetic predictor...")
    aesthetic_model = AutoModel.from_pretrained(
        AESTHETICS_REPO,
        trust_remote_code=True,
    ).to("cuda")
    aesthetic_processor = AutoProcessor.from_pretrained(
        AESTHETICS_REPO,
        trust_remote_code=True,
    )
    _log("[init] Aesthetic predictor ready.")


def calc_score(img: Image.Image) -> float:
    assert aesthetic_model is not None and aesthetic_processor is not None
    inputs = aesthetic_processor(images=img, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        outputs = aesthetic_model(**inputs)
    score = float(outputs.logits.reshape(-1)[0].item())
    return round(score, 2)


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

    score = calc_score(img)

    return {
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
        if job.get("input") == "health":
            return {
                "status": "healthy",
                "pipe_loaded": pipe is not None,
                "aesthetics_loaded": aesthetic_model is not None,
                "gpu_available": torch.cuda.is_available(),
                "hf_home": os.environ.get("HF_HOME"),
            }

        inp = job["input"]
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


_log("[boot] starting runpod serverless loop...")
runpod.serverless.start({"handler": handler})
