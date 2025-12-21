"""
RunPod Serverless Handler - Unified Endpoint with /create and /production routes
Uses Playground v2.5 with proper EDM schedulers and aesthetic-predictor-v2-5
"""

import runpod
import torch
from diffusers import DiffusionPipeline, EDMDPMSolverMultistepScheduler, EDMEulerScheduler
from PIL import Image
import io
import base64
from transformers import AutoModel, AutoProcessor

# Global variables for model persistence
pipe = None
aesthetic_model = None
aesthetic_processor = None

def load_models():
    """Load all models once during cold start"""
    global pipe, aesthetic_model, aesthetic_processor
    
    if pipe is None:
        print("Loading Playground v2.5 model...")
        pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2.5-1024px-aesthetic",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        
        # Use DPM++ 2M Karras scheduler (EDM formulation) as recommended
        pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        print("✓ Playground v2.5 loaded with EDMDPMSolverMultistepScheduler")
    
    if aesthetic_model is None:
        print("Loading aesthetic predictor v2.5 (SigLIP-based)...")
        aesthetic_model = AutoModel.from_pretrained(
            "discus0434/aesthetic-predictor-v2-5",
            trust_remote_code=True
        ).to("cuda")
        aesthetic_processor = AutoProcessor.from_pretrained(
            "discus0434/aesthetic-predictor-v2-5",
            trust_remote_code=True
        )
        print("✓ Aesthetic predictor v2.5 loaded")

def calculate_aesthetic_score(image):
    """Calculate aesthetic score using the v2.5 predictor"""
    global aesthetic_model, aesthetic_processor
    
    inputs = aesthetic_processor(images=image, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = aesthetic_model(**inputs)
        # The model returns a score directly
        score = outputs.logits.item()
    
    return round(score, 2)

def route_create(job_input):
    """
    /create route - Fast image generation with aesthetic scoring
    Uses DPM++ 2M Karras (guidance_scale=3.0 recommended)
    
    Expected input:
    {
        "route": "create",
        "prompt": "your positive prompt",
        "negative_prompt": "your negative prompt (optional)",
        "width": 1024,
        "height": 1024,
        "seed": -1,
        "steps": 50,
        "guidance_scale": 3.0,
        "scheduler": "dpm",  # "dpm" or "euler"
        "filename": "custom_filename (optional)"
    }
    """
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    seed = job_input.get("seed", -1)
    steps = job_input.get("steps", 50)
    guidance_scale = job_input.get("guidance_scale", 3.0)
    scheduler_type = job_input.get("scheduler", "dpm")
    filename = job_input.get("filename", prompt[:50])
    
    # Set scheduler based on request
    if scheduler_type == "euler":
        pipe.scheduler = EDMEulerScheduler.from_config(pipe.scheduler.config)
        if guidance_scale == 3.0:  # If using default, adjust for Euler
            guidance_scale = 5.0
        print(f"[CREATE] Using EDMEulerScheduler with guidance_scale={guidance_scale}")
    else:
        pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print(f"[CREATE] Using EDMDPMSolverMultistepScheduler with guidance_scale={guidance_scale}")
    
    # Handle random seed
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print(f"[CREATE] Generating: {width}x{height}, seed={seed}, steps={steps}")
    
    # Generate image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    
    # Calculate aesthetic score
    aesthetic_score = calculate_aesthetic_score(image)
    print(f"[CREATE] Aesthetic score: {aesthetic_score}")
    
    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "route": "create",
        "image": img_base64,
        "aesthetic_score": aesthetic_score,
        "metadata": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler_type,
            "filename": filename,
            "model": "playground-v2.5-1024px-aesthetic"
        }
    }

def route_production(job_input):
    """
    /production route - High-quality generation (upscaling disabled for now)
    Uses higher steps and can use either scheduler
    
    Expected input:
    {
        "route": "production",
        "prompt": "your positive prompt",
        "negative_prompt": "your negative prompt (optional)",
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "steps": 75,
        "guidance_scale": 3.0,
        "scheduler": "dpm"
    }
    """
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    seed = job_input.get("seed", 0)
    steps = job_input.get("steps", 75)
    guidance_scale = job_input.get("guidance_scale", 3.0)
    scheduler_type = job_input.get("scheduler", "dpm")
    
    # Set scheduler based on request
    if scheduler_type == "euler":
        pipe.scheduler = EDMEulerScheduler.from_config(pipe.scheduler.config)
        if guidance_scale == 3.0:
            guidance_scale = 5.0
        print(f"[PRODUCTION] Using EDMEulerScheduler with guidance_scale={guidance_scale}")
    else:
        pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print(f"[PRODUCTION] Using EDMDPMSolverMultistepScheduler with guidance_scale={guidance_scale}")
    
    # Handle random seed
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print(f"[PRODUCTION] Generating: {width}x{height}, seed={seed}, steps={steps}")
    
    # Generate high-quality image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    
    # Calculate aesthetic score
    aesthetic_score = calculate_aesthetic_score(image)
    print(f"[PRODUCTION] Aesthetic score: {aesthetic_score}")
    
    # Note: Upscaling disabled for now, can be added later
    
    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "route": "production",
        "image": img_base64,
        "aesthetic_score": aesthetic_score,
        "metadata": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler_type,
            "model": "playground-v2.5-1024px-aesthetic",
            "note": "Upscaling disabled, will be added in future update"
        }
    }

def handler(job):
    """
    Main handler - routes to /create or /production based on input
    
    Input must include a "route" field with value "create" or "production"
    """
    try:
        job_input = job["input"]
        route = job_input.get("route", "").lower()
        
        # Load models on first request
        load_models()
        
        if route == "create":
            return route_create(job_input)
        elif route == "production":
            return route_production(job_input)
        else:
            return {
                "error": f"Invalid route: '{route}'. Must be 'create' or 'production'",
                "valid_routes": ["create", "production"]
            }
            
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
