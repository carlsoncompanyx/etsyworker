# handler.py  (create: 25 steps + cfg 7 + random seed + score; production: 50 steps + cfg 7 + seed required + NO score)
from __future__ import annotations

import base64
import io
import os
import sys
import traceback
from typing import Any, Dict, Optional

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

LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"

pipe: Optional[DiffusionPipeline] = None
aesthetic_model: Optional[torch.nn.Module] = None
aesthetic_processor: Optional[Any] = None


def _log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def load_pipe() -> None:
    global pipe
    if pipe is not None:
        return

    _log("[init] Loading Playground v2.5...")
    pipe = DiffusionPipeline.from_pretrained(
        PLAYGROUND_REPO,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=LOCAL_FILES_ONLY,
    ).to("cuda")
    pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    _log("[init] Playground ready")


def load_aesthetics() -> None:
    global aesthetic_model, aesthetic_processor
    if aesthetic_model is not None and aesthetic_processor is not None:
        return

    _log("[init] Loading aesthetic predictor...")
    aesthetic_model = AutoModel.from_pretrained(
        AESTHETICS_REPO,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    ).to("cuda")
    aesthetic_processor = AutoProcessor.from_pretrained(
        AESTHETICS_REPO,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    _log("[init] Aesthetic predictor ready")


def _img_to_b64_jpeg(img: Image.Image, quality: int = 95) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _calc_score(img: Image.Image) -> float:
    assert aesthetic_model is not None and aesthetic_processor is not None
    inputs = aesthetic_processor(images=img, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        outputs = aesthetic_model(**inputs)

    if hasattr(outputs, "logits") and outputs.logits is not None:
        score = float(outputs.logits.reshape(-1)[0].item())
    elif isinstance(outputs, (tuple, list)) and outputs:
        score = float(torch.as_tensor(outputs[0]).reshape(-1)[0].item())
    else:
        raise RuntimeError("Aesthetic model returned unexpected output format")

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

    score = _calc_score(img)

    return {
        "route": "create",
        "image": _img_to_b64_jpeg(img),
        "aesthetic_score": score,
        "metadata": {
            "prompt": prompt,
            "negative_prompt": negative,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": CREATE_STEPS,
            "cfg_scale": CFG,
            "model": "playground-v2.5-1024px-aesthetic",
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
        raise ValueError("production requires 'seed' (reuse from create)")

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
            "prompt": prompt,
            "negative_prompt": negative,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": PRODUCTION_STEPS,
            "cfg_scale": CFG,
            "model": "playground-v2.5-1024px-aesthetic",
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
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
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


if __name__ == "__main__":
    _log("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
