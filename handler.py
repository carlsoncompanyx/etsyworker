"""
RunPod Serverless Handler - Unified Endpoint with /create and /production routes
Replicates both ComfyUI workflows in a single endpoint
"""

import runpod
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
import io
import base64
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import math

# Global variables for model persistence
pipe = None
aesthetic_model = None
aesthetic_processor = None
upscaler = None

def load_models():
    """Load all models once during cold start"""
    global pipe, aesthetic_model, aesthetic_processor, upscaler
    
    if pipe is None:
        print("Loading Playground v2.5 model...")
        pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2.5-1024px-aesthetic",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe = pipe.to("cuda")
        
        # Configure scheduler to match ComfyUI settings
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing"
        )
        
        print("Playground model loaded successfully")
    
    if aesthetic_model is None:
        print("Loading aesthetic predictor...")
        aesthetic_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        aesthetic_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        aesthetic_model = aesthetic_model.to("cuda")
        print("Aesthetic model loaded successfully")
    
    if upscaler is None:
        print("Loading Real-ESRGAN 4x upscaler...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upscaler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device='cuda'
        )
        print("Upscaler loaded successfully")

def calculate_aesthetic_score(image):
    """Calculate aesthetic score for the generated image"""
    global aesthetic_model, aesthetic_processor
    
    inputs = aesthetic_processor(images=image, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        image_features = aesthetic_model.get_image_features(**inputs)
        score = torch.norm(image_features).item() / 100.0
        score = min(max(score, 0.0), 10.0)
    
    return round(score, 2)

def tile_upscale(image, tile_size=1024, overlap=64):
    """Perform tiled upscaling for large images (UltimateSDUpscale logic)"""
    global upscaler
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Calculate output dimensions
    out_h, out_w = h * 4, w * 4
    output = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    # Calculate tiles
    tiles_x = math.ceil(w / (tile_size - overlap))
    tiles_y = math.ceil(h / (tile_size - overlap))
    
    print(f"Tiled upscaling: {tiles_x}x{tiles_y} tiles")
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x1 = tx * (tile_size - overlap)
            y1 = ty * (tile_size - overlap)
            x2 = min(x1 + tile_size, w)
            y2 = min(y1 + tile_size, h)
            
            tile = img_array[y1:y2, x1:x2]
            upscaled_tile, _ = upscaler.enhance(tile, outscale=4)
            
            out_x1, out_y1 = x1 * 4, y1 * 4
            out_x2, out_y2 = x2 * 4, y2 * 4
            
            output[out_y1:out_y2, out_x1:out_x2] = upscaled_tile
    
    return Image.fromarray(output)

def route_create(job_input):
    """
    /create route - Flow A: Fast image generation with aesthetic scoring
    
    Expected input:
    {
        "route": "create",
        "prompt": "your positive prompt",
        "negative_prompt": "your negative prompt (optional)",
        "width": 1152,
        "height": 768,
        "seed": -1,
        "steps": 25,
        "cfg_scale": 7.0,
        "filename": "custom_filename (optional)"
    }
    """
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")
    width = job_input.get("width", 1152)
    height = job_input.get("height", 768)
    seed = job_input.get("seed", -1)
    steps = job_input.get("steps", 25)
    cfg_scale = job_input.get("cfg_scale", 7.0)
    filename = job_input.get("filename", prompt[:50])
    
    # Handle random seed
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print(f"[CREATE] Generating: {width}x{height}, seed={seed}, steps={steps}, cfg={cfg_scale}")
    
    # Generate image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
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
            "cfg_scale": cfg_scale,
            "filename": filename,
            "model": "playground-v2.5-1024px-aesthetic"
        }
    }

def route_production(job_input):
    """
    /production route - Flow B: High-quality generation with 4x upscaling
    
    Expected input:
    {
        "route": "production",
        "prompt": "your positive prompt",
        "negative_prompt": "your negative prompt (optional)",
        "width": 1024,
        "height": 680,
        "seed": 0,
        "steps": 50,
        "cfg_scale": 7.0,
        "upscale_factor": 4.0,
        "use_tiled_upscale": true,
        "tile_size": 1024
    }
    """
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 680)
    seed = job_input.get("seed", 0)
    steps = job_input.get("steps", 50)
    cfg_scale = job_input.get("cfg_scale", 7.0)
    upscale_factor = job_input.get("upscale_factor", 4.0)
    use_tiled = job_input.get("use_tiled_upscale", True)
    tile_size = job_input.get("tile_size", 1024)
    
    # Handle random seed
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print(f"[PRODUCTION] Generating: {width}x{height}, seed={seed}, steps={steps}, cfg={cfg_scale}")
    
    # Generate base image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator
    ).images[0]
    
    print("[PRODUCTION] Base image generated, starting upscaling...")
    
    # First pass: 4x upscaling with Real-ESRGAN
    img_array = np.array(image)
    upscaled_img, _ = upscaler.enhance(img_array, outscale=upscale_factor)
    upscaled_image = Image.fromarray(upscaled_img)
    
    print(f"[PRODUCTION] First pass complete: {upscaled_image.size}")
    
    # Optional tiled upscaling for refinement
    if use_tiled:
        print("[PRODUCTION] Applying tiled upscaling refinement...")
        final_image = tile_upscale(upscaled_image, tile_size=tile_size)
    else:
        final_image = upscaled_image
    
    print(f"[PRODUCTION] Final size: {final_image.size}")
    
    # Convert to base64
    buffered = io.BytesIO()
    final_image.save(buffered, format="JPEG", quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "route": "production",
        "image": img_base64,
        "metadata": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "base_width": width,
            "base_height": height,
            "final_width": final_image.size[0],
            "final_height": final_image.size[1],
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "upscale_factor": upscale_factor,
            "model": "playground-v2.5-1024px-aesthetic",
            "upscaler": "Real-ESRGAN 4x"
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
