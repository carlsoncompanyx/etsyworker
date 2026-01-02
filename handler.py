# handler.py
from __future__ import annotations
import base64
import io
import os
import sys
import traceback
from typing import Any, Dict, Optional

# Add the GitHub folder to path so we can import the predictor logic
sys.path.append('/workspace/aesthetic-predictor-v2-5')

import runpod
import torch
import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EDMDPMSolverMultistepScheduler
from transformers import AutoModel, AutoProcessor

# --- PATH CONFIGURATION ---
# These point to the exact locations we verified on your /workspace volume
MODEL_PATH = "/workspace/playground-v2.5-1024px-aesthetic.fp16.safetensors"
SIGLIP_PATH = "/workspace/siglip"
PREDICTOR_WEIGHTS = "/workspace/aesthetic-predictor-v2-5/aesthetic-predictor.pth"

CFG = 7.0
CREATE_STEPS = 25
PRODUCTION_STEPS = 50

# Global model variables
pipe: Optional[StableDiffusionXLPipeline] = None
aesthetic_model: Optional[torch.nn.Module] = None
aesthetic_processor: Optional[Any] = None

def _log(msg: str) -> None:
    print(msg, flush=True)

def _img_to_b64_jpeg(img: Image.Image, quality: int = 95) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def load_models() -> None:
    global pipe, aesthetic_model, aesthetic_processor
    
    # 1. Load Playground v2.5 from single safetensors file
    if pipe is None:
        _log(f"[init] loading playground v2.5 from {MODEL_PATH}...")
        pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_PATH,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        _log("[init] pipeline ready")

    # 2. Load SigLIP Backbone and Aesthetic Predictor
    if aesthetic_model is None:
        _log(f"[init] loading aesthetic brain from {SIGLIP_PATH}...")
        # Import local class from cloned github repo
        from aesthetic_predictor_v2_5 import AestheticPredictorV2_5
        
        backbone = AutoModel.from_pretrained(SIGLIP_PATH).to("cuda")
        aesthetic_processor = AutoProcessor.from_pretrained(SIGLIP_PATH)
        
        # Initialize head and load specific .pth weights
        aesthetic_model = AestheticPredictorV2_5(backbone).to("cuda")
        aesthetic_model.load_state_dict(torch.load(PREDICTOR_WEIGHTS))
        aesthetic_model.eval()
        _log("[init] aesthetic predictor ready")

def _calc_score(img: Image.Image) -> float:
    inputs = aesthetic_processor(images=img, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        score = aesthetic_model(inputs.pixel_values)
    return round(float(score.item()), 2)

def route_create(inp: Dict[str, Any]) -> Dict[str, Any]:
    load_models()
    assert pipe is not None

    prompt = str(inp.get("prompt", ""))
    negative = str(inp.get("negative_prompt", ""))
    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))

    seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    gen = torch.Generator(device="cuda").manual_seed(seed)

    _log(f"[create] {width}x{height} seed={seed} steps={CREATE_STEPS}")

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
        "metadata": {"seed": seed, "steps": CREATE_STEPS, "width": width, "height": height}
    }

def route_production(inp: Dict[str, Any]) -> Dict[str, Any]:
    load_models()
    assert pipe is not None

    prompt = str(inp.get("prompt", ""))
    negative = str(inp.get("negative_prompt", ""))
    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))

    if "seed" not in inp:
        raise ValueError("production requires 'seed' from create")

    seed = int(inp["seed"])
    gen = torch.Generator(device="cuda").manual_seed(seed)

    _log(f"[production] {width}x{height} seed={seed} steps={PRODUCTION_STEPS}")

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
        "metadata": {"seed": seed, "steps": PRODUCTION_STEPS, "width": width, "height": height}
    }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input")
        if inp == "health":
            return {"status": "healthy", "gpu": torch.cuda.get_device_name(0)}

        assert isinstance(inp, dict)
        route = str(inp.get("route", "")).lower()

        if route == "create":
            return route_create(inp)
        if route == "production":
            return route_production(inp)

        return {"error": "route must be 'create' or 'production'"}
    except Exception as e:
        _log(f"[ERR] {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
