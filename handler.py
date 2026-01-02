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
if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import file_download
    def _cached_download(*args, **kwargs):
        return file_download.hf_hub_download(*args, **kwargs)
    huggingface_hub.cached_download = _cached_download

from diffusers import StableDiffusionXLPipeline, EDMDPMSolverMultistepScheduler
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
        try:
            # Load the pipeline
            pipe = StableDiffusionXLPipeline.from_single_file(
                MODEL_PATH,
                torch_dtype=torch.float16,
                use_safetensors=True,
                load_safety_checker=False
            ).to("cuda")
            
            # Use EDMDPMSolverMultistepScheduler as recommended by Playground
            pipe.scheduler = EDMDPMSolverMultistepScheduler()
            
            # Enable memory optimizations
            pipe.enable_attention_slicing()
            
            print("Pipeline loaded successfully")
            print(f"Scheduler: {type(pipe.scheduler).__name__}")
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise

    if aesthetic_model is None:
        print(f"Loading Predictor from {SIGLIP_PATH}")
        try:
            from aesthetic_predictor_v2_5 import AestheticPredictorV2_5
            
            backbone = AutoModel.from_pretrained(SIGLIP_PATH).to("cuda")
            aesthetic_processor = AutoProcessor.from_pretrained(SIGLIP_PATH)
            
            aesthetic_model = AestheticPredictorV2_5(backbone).to("cuda")
            aesthetic_model.load_state_dict(torch.load(PREDICTOR_WEIGHTS, map_location="cuda"))
            aesthetic_model.eval()
            print("Aesthetic model loaded successfully")
        except Exception as e:
            print(f"Error loading aesthetic model: {e}")
            raise

def handler(job):
    try:
        load_models()
        inp = job.get("input", {})
        route = inp.get("route", "create")
        prompt = inp.get("prompt", "A high quality photo")
        seed = inp.get("seed", None)
        
        # Route-specific settings
        if route == "create":
            # Fast route: fewer steps
            num_steps = 25
            guidance_scale = 3.0
        elif route == "production":
            # High quality route: more steps
            num_steps = 50
            guidance_scale = 3.0
        else:
            # Default/custom settings
            num_steps = inp.get("num_inference_steps", 25)
            guidance_scale = inp.get("guidance_scale", 3.0)
        
        # Set seed if provided, otherwise generate one
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        print(f"Route: {route}")
        print(f"Generating image with prompt: {prompt}")
        print(f"Steps: {num_steps}, Guidance: {guidance_scale}, Seed: {seed}")
        
        # Generation logic
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        print("Image generated, calculating aesthetic score...")
        
        # Scoring logic
        inputs = aesthetic_processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            score = aesthetic_model(inputs.pixel_values)
        
        # Encode
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        result = {
            "image": img_str,
            "aesthetic_score": round(float(score.item()), 2),
            "seed": seed
        }
        
        print(f"Success! Aesthetic score: {result['aesthetic_score']}")
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}

if __name__ == "__main__":
    print("Starting RunPod handler...")
    runpod.serverless.start({"handler": handler})
