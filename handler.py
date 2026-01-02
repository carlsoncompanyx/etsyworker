import os
import sys
import base64
import io
import torch
import torch.nn as nn
from PIL import Image
import runpod
import huggingface_hub

# Backward-compatibility shim for diffusers<=0.25 expecting cached_download
# huggingface_hub 0.23+ removed cached_download; remap to hf_hub_download when missing
if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import file_download

    def _cached_download(*args, **kwargs):
        return file_download.hf_hub_download(*args, **kwargs)

    huggingface_hub.cached_download = _cached_download

import diffusers
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# Import scheduler via attribute lookup to avoid hard import errors in older diffusers
SchedulerCls = getattr(diffusers, "EDMDPMSolverMultistepScheduler", DPMSolverMultistepScheduler)
from transformers import AutoModel, AutoProcessor

# Add GitHub repo to path
sys.path.append('/workspace/aesthetic-predictor-v2-5')

# Paths verified on your volume
MODEL_PATH = "/workspace/playground-v2.5-1024px-aesthetic.fp16.safetensors"
SIGLIP_PATH = "/workspace/siglip"
PREDICTOR_WEIGHTS = "/workspace/aesthetic-predictor-v2-5/aesthetic-predictor.pth"

pipe = None
aesthetic_model = None
aesthetic_processor = None

def load_models():
    global pipe, aesthetic_model, aesthetic_processor
    
    if pipe is None:
        print(f"Loading Playground from {MODEL_PATH}")
        pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_PATH,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        pipe.scheduler = SchedulerCls.from_config(pipe.scheduler.config)

    if aesthetic_model is None:
        print(f"Loading Predictor from {SIGLIP_PATH}")
        from aesthetic_predictor_v2_5 import AestheticPredictorV2_5
        
        backbone = AutoModel.from_pretrained(SIGLIP_PATH).to("cuda")
        aesthetic_processor = AutoProcessor.from_pretrained(SIGLIP_PATH)
        
        aesthetic_model = AestheticPredictorV2_5(backbone).to("cuda")
        aesthetic_model.load_state_dict(torch.load(PREDICTOR_WEIGHTS))
        aesthetic_model.eval()

def handler(job):
    try:
        load_models()
        inp = job.get("input")
        prompt = inp.get("prompt", "A high quality photo")
        
        # Generation logic
        image = pipe(prompt=prompt, num_inference_steps=25).images[0]
        
        # Scoring logic
        inputs = aesthetic_processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            score = aesthetic_model(inputs.pixel_values)
        
        # Encode
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "image": img_str,
            "aesthetic_score": round(float(score.item()), 2)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
