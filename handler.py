"""
RunPod Serverless GPU Worker for Etsy Store Automation
Handles: Image Generation (Playground v2.5), Aesthetic Scoring (LAION), Upscaling (Real-ESRGAN)

Supports three process modes:
- initial: Generate images, score them, optionally filter by top_k
- recreate: Generate high-quality images (50 steps), upscale, no scoring
- upscale: Upscale user-supplied images to target resolution

Author: Your Name
Version: 1.1.0
"""

import runpod
import torch
import base64
import io
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from diffusers import DiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np

# ============================================================================
# GLOBAL MODEL LOADING (occurs once per worker initialization)
# ============================================================================

print("🚀 Initializing GPU models...")

# Playground v2.5 1024px Aesthetic Model
PLAYGROUND_MODEL_ID = "playgroundai/playground-v2.5-1024px-aesthetic"
print(f"📦 Loading Playground v2.5 from {PLAYGROUND_MODEL_ID}...")
playground_pipe = DiffusionPipeline.from_pretrained(
    PLAYGROUND_MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
playground_pipe.enable_attention_slicing()
print("✅ Playground v2.5 loaded")

# LAION Aesthetic Predictor (for scoring)
AESTHETIC_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
print(f"📦 Loading LAION aesthetic model from {AESTHETIC_MODEL_ID}...")
aesthetic_model = CLIPModel.from_pretrained(AESTHETIC_MODEL_ID).to("cuda")
aesthetic_processor = CLIPProcessor.from_pretrained(AESTHETIC_MODEL_ID)
print("✅ LAION aesthetic model loaded")

# Real-ESRGAN Upscalers (multiple models for different content types)
print("📦 Loading Real-ESRGAN models...")
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import os
    import urllib.request
    
    # Global upsampler dictionary
    upsamplers = {}
    
    # Model definitions with download URLs
    models_config = {
        'photo': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'path': '/app/models/RealESRGAN_x4plus.pth',
            'arch': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 23, 'num_grow_ch': 32, 'scale': 4}
        },
        'art': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_anime_6B.pth',
            'path': '/app/models/RealESRGAN_x4plus_anime_6B.pth',
            'arch': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 6, 'num_grow_ch': 32, 'scale': 4}
        },
        'conservative': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
            'path': '/app/models/RealESRNet_x4plus.pth',
            'arch': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 23, 'num_grow_ch': 32, 'scale': 4}
        }
    }
    
    # Download and load each model
    for model_name, config in models_config.items():
        print(f"   Loading {model_name} upscaler...")
        
        # Download model if it doesn't exist
        if not os.path.exists(config['path']):
            print(f"   Downloading {model_name} model from {config['url']}...")
            try:
                urllib.request.urlretrieve(config['url'], config['path'])
                print(f"   ✅ Downloaded {model_name} model")
            except Exception as e:
                print(f"   ⚠️ Failed to download {model_name} model: {e}")
                continue
        
        # Load the model
        try:
            model = RRDBNet(**config['arch'])
            upsamplers[model_name] = RealESRGANer(
                scale=4,
                model_path=config['path'],
                model=model,
                tile=512,
                tile_pad=10,
                pre_pad=0,
                half=True,
                gpu_id=0
            )
            print(f"   ✅ {model_name} upscaler ready")
        except Exception as e:
            print(f"   ⚠️ Failed to load {model_name} upscaler: {e}")
    
    print(f"✅ {len(upsamplers)}/{len(models_config)} Real-ESRGAN models loaded")
except Exception as e:
    print(f"⚠️ Real-ESRGAN failed to initialize: {e}")
    upsamplers = {}

print("🎉 All models loaded successfully\n")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def b64_to_pil(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def pil_to_b64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def score_image_aesthetic(image: Image.Image) -> float:
    """
    Score an image using LAION aesthetic predictor.
    Returns a float score (typically 0-10, higher is better).
    """
    try:
        inputs = aesthetic_processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            image_features = aesthetic_model.get_image_features(**inputs)
            # Normalize and get aesthetic score
            # Note: This is a simplified version - you may need to adjust based on your specific model
            score = image_features.norm().item()
        return float(score)
    except Exception as e:
        print(f"⚠️ Scoring failed: {e}")
        return 0.0


def upscale_image_realesrgan(
    image: Image.Image, 
    target_resolution: Optional[Tuple[int, int]] = None,
    model_type: str = "art"
) -> Image.Image:
    """
    Upscale image using Real-ESRGAN.
    
    Args:
        image: Input PIL Image
        target_resolution: Optional (width, height) tuple. If provided, performs multiple 
                          4x upscale passes until target is met or exceeded.
        model_type: Which upscaler to use:
                   - "photo": RealESRGAN_x4plus (best for photorealistic images)
                   - "art": RealESRGAN_x4plus_anime_6B (best for artwork, watercolors, paintings)
                   - "conservative": RealESRNet_x4plus (less aggressive, fewer artifacts)
    
    Returns:
        Upscaled PIL Image
    """
    if not upsamplers:
        raise RuntimeError("Real-ESRGAN upsamplers not initialized")
    
    if model_type not in upsamplers:
        print(f"⚠️ Model type '{model_type}' not found, defaulting to 'art'")
        model_type = "art"
    
    upsampler = upsamplers[model_type]
    print(f"🎨 Using upscaler: {model_type}")
    
    current_img = image
    pass_count = 0
    max_passes = 5  # Safety limit
    
    while pass_count < max_passes:
        pass_count += 1
        current_width, current_height = current_img.size
        
        # Check if we've met target resolution
        if target_resolution:
            target_w, target_h = target_resolution
            if current_width >= target_w and current_height >= target_h:
                print(f"✅ Target resolution reached: {current_width}x{current_height}")
                break
        
        print(f"🔄 Upscale pass {pass_count}: {current_width}x{current_height} → ", end="")
        
        # Convert PIL to numpy array for Real-ESRGAN
        img_array = np.array(current_img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Perform upscaling
        try:
            upscaled_array, _ = upsampler.enhance(img_array, outscale=4)
            upscaled_array = cv2.cvtColor(upscaled_array, cv2.COLOR_BGR2RGB)
            current_img = Image.fromarray(upscaled_array)
            new_width, new_height = current_img.size
            print(f"{new_width}x{new_height}")
        except Exception as e:
            print(f"\n⚠️ Upscale pass {pass_count} failed: {e}")
            break
        
        # If no target resolution specified, do single pass and exit
        if not target_resolution:
            break
    
    return current_img


# ============================================================================
# PROCESS MODE HANDLERS
# ============================================================================

def process_initial(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    PROCESS A: Generate images, score them, optionally filter by top_k.
    
    No upscaling in this mode.
    """
    print("\n📸 PROCESS: initial (generate + score)")
    
    prompts = job_input.get("prompts", [])
    negative_prompt = job_input.get("negative_prompt", "")
    num_images = job_input.get("num_images", 1)
    seed = job_input.get("seed", None)
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    steps = job_input.get("steps", 25)
    guidance_scale = job_input.get("guidance_scale", 7.5)
    top_k = job_input.get("top_k", None)
    
    if not prompts:
        raise ValueError("prompts list is required for initial process")
    
    results = []
    
    for idx, prompt in enumerate(prompts):
        print(f"\n🎨 Generating {num_images} images for prompt {idx+1}/{len(prompts)}")
        print(f"   Prompt: {prompt[:80]}...")
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed + idx)
        
        # Generate images
        output = playground_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        # Score each image
        for img_idx, pil_image in enumerate(output.images):
            image_id = f"img_{idx:03d}_{img_idx:02d}"
            
            # Score image
            score = score_image_aesthetic(pil_image)
            print(f"   {image_id}: score = {score:.4f}")
            
            results.append({
                "id": image_id,
                "b64": pil_to_b64(pil_image),
                "prompt": prompt,
                "score": score,
                "width": pil_image.width,
                "height": pil_image.height,
                "stored_path": None
            })
    
    # Apply top_k filtering if requested
    if top_k and top_k < len(results):
        print(f"\n🔝 Filtering to top {top_k} images by score")
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
    
    return {
        "images": results,
        "upscaled_images": []
    }


def process_recreate(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    PROCESS B: Generate high-quality images (50 steps), then upscale them.
    
    No scoring in this mode.
    """
    print("\n🎯 PROCESS: recreate (generate high-quality + upscale)")
    
    prompts = job_input.get("prompts", [])
    negative_prompt = job_input.get("negative_prompt", "")
    num_images = job_input.get("num_images", 1)
    seed = job_input.get("seed", None)
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    steps = job_input.get("steps", 50)
    guidance_scale = job_input.get("guidance_scale", 7.5)
    final_resolution = job_input.get("final_resolution", None)
    upscale_model = job_input.get("upscale_model", "art")  # NEW: Allow model selection
    
    if not prompts:
        raise ValueError("prompts list is required for recreate process")
    
    generated_images = []
    upscaled_images = []
    
    for idx, prompt in enumerate(prompts):
        print(f"\n🎨 Generating {num_images} high-quality images for prompt {idx+1}/{len(prompts)}")
        print(f"   Prompt: {prompt[:80]}...")
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed + idx)
        
        # Generate images
        output = playground_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        # Store generated images and upscale each
        for img_idx, pil_image in enumerate(output.images):
            image_id = f"img_{idx:03d}_{img_idx:02d}"
            
            # Store original generated image
            generated_images.append({
                "id": image_id,
                "b64": pil_to_b64(pil_image),
                "prompt": prompt,
                "score": None,
                "width": pil_image.width,
                "height": pil_image.height,
                "stored_path": None
            })
            
            # Upscale
            print(f"   🔼 Upscaling {image_id}...")
            upscaled = upscale_image_realesrgan(
                pil_image, 
                target_resolution=final_resolution,
                model_type=upscale_model
            )
            
            upscaled_images.append({
                "id": f"{image_id}_up",
                "source_id": image_id,
                "b64": pil_to_b64(upscaled),
                "width": upscaled.width,
                "height": upscaled.height,
                "stored_path": None
            })
    
    return {
        "images": generated_images,
        "upscaled_images": upscaled_images
    }


def process_upscale(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    PROCESS C: Upscale user-supplied images.
    
    No generation or scoring.
    """
    print("\n🔼 PROCESS: upscale (upscale only)")
    
    images_b64 = job_input.get("images_b64", [])
    final_resolution = job_input.get("final_resolution", None)
    upscale_model = job_input.get("upscale_model", "art")  # NEW: Allow model selection
    
    if not images_b64:
        raise ValueError("images_b64 list is required for upscale process")
    
    upscaled_images = []
    
    for idx, b64_str in enumerate(images_b64):
        image_id = f"img_{idx:03d}"
        print(f"\n🔼 Upscaling image {idx+1}/{len(images_b64)}")
        
        # Decode image
        pil_image = b64_to_pil(b64_str)
        print(f"   Input size: {pil_image.width}x{pil_image.height}")
        
        # Upscale
        upscaled = upscale_image_realesrgan(
            pil_image, 
            target_resolution=final_resolution,
            model_type=upscale_model
        )
        
        upscaled_images.append({
            "id": f"{image_id}_up",
            "source_id": image_id,
            "b64": pil_to_b64(upscaled),
            "width": upscaled.width,
            "height": upscaled.height,
            "stored_path": None
        })
    
    return {
        "images": [],
        "upscaled_images": upscaled_images
    }


# ============================================================================
# MAIN HANDLER
# ============================================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler function.
    
    Routes to appropriate process mode and returns standardized response.
    """
    start_time = time.time()
    job_input = job.get("input", {})
    
    try:
        # Extract process mode
        process_mode = job_input.get("process", "initial")
        job_id = job_input.get("meta", {}).get("job_id", "unknown")
        
        print(f"\n{'='*80}")
        print(f"🚀 NEW JOB: {job_id}")
        print(f"📋 Process mode: {process_mode}")
        print(f"{'='*80}")
        
        # Route to appropriate handler
        if process_mode == "initial":
            result = process_initial(job_input)
        elif process_mode == "recreate":
            result = process_recreate(job_input)
        elif process_mode == "upscale":
            result = process_upscale(job_input)
        else:
            raise ValueError(f"Unknown process mode: {process_mode}")
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Build response
        response = {
            "process": process_mode,
            "images": result["images"],
            "upscaled_images": result["upscaled_images"],
            "meta": {
                "job_id": job_id,
                "duration_ms": duration_ms
            },
            "status": "ok"
        }
        
        print(f"\n✅ Job complete in {duration_ms}ms")
        print(f"   Generated images: {len(result['images'])}")
        print(f"   Upscaled images: {len(result['upscaled_images'])}")
        
        return response
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_trace = traceback.format_exc()
        
        print(f"\n❌ Job failed: {str(e)}")
        print(f"\n{error_trace}")
        
        return {
            "status": "error",
            "error": str(e),
            "trace": error_trace,
            "meta": {
                "job_id": job_input.get("meta", {}).get("job_id", "unknown"),
                "duration_ms": duration_ms
            }
        }


# ============================================================================
# RUNPOD INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎉 RunPod Serverless Worker Ready")
    print("="*80 + "\n")
    runpod.serverless.start({"handler": handler})
