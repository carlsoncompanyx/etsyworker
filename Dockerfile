"""
RunPod Serverless Handler - Unified Image Generation + Upscaling
Handles both: prompt -> generate -> upscale, and image -> upscale only
"""

import runpod
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import io
import base64
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLAYGROUND_MODEL = "playgroundai/playground-v2.5-1024px-aesthetic"

# Upscaler models available
UPSCALER_MODELS = {
    "realesrgan_x4": "RealESRGAN_x4plus.pth",
    "realesrgan_anime": "RealESRGAN_x4plus_anime_6B.pth",
    "realesrgan_x2": "RealESRGAN_x2plus.pth"
}

# ============================================================================
# GLOBAL MODEL LOADING (on container startup)
# ============================================================================

print("🔥 Loading Playground v2.5...")
pipe = DiffusionPipeline.from_pretrained(
    PLAYGROUND_MODEL,
    torch_dtype=torch.float16,
    variant="fp16"
).to(DEVICE)

# Enable optimizations
pipe.enable_attention_slicing()

# Load default upscaler (x4plus for general use)
print("🔥 Loading Real-ESRGAN upscaler...")
upscaler_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
upscaler = RealESRGANer(
    scale=4,
    model_path=f"weights/{UPSCALER_MODELS['realesrgan_x4']}",
    model=upscaler_model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True,
    device=DEVICE
)

print("✅ Models loaded successfully!")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def decode_base64_image(base64_string):
    """Decode base64 image to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def encode_image_to_base64(image):
    """Encode PIL Image to base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def download_image_from_url(url):
    """Download image from URL"""
    response = requests.get(url, timeout=30)
    return Image.open(io.BytesIO(response.content))

def upscale_image(image, target_dpi=300, upscale_model="realesrgan_x4"):
    """
    Upscale image to target DPI using Real-ESRGAN
    
    Args:
        image: PIL Image
        target_dpi: Target DPI (default 300 for print)
        upscale_model: Which upscaler to use
    """
    global upscaler
    
    # Switch upscaler if different model requested
    if upscale_model != "realesrgan_x4":
        print(f"🔄 Switching to {upscale_model}...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        upscaler = RealESRGANer(
            scale=4,
            model_path=f"weights/{UPSCALER_MODELS[upscale_model]}",
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=DEVICE
        )
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Upscale
    output, _ = upscaler.enhance(img_array, outscale=4)
    
    # Convert back to PIL
    upscaled_image = Image.fromarray(output)
    
    # Set DPI metadata
    upscaled_image.info['dpi'] = (target_dpi, target_dpi)
    
    return upscaled_image

def generate_image(prompt, negative_prompt="", steps=50, guidance_scale=3.0, seed=None):
    """
    Generate image using Playground v2.5
    
    Args:
        prompt: Text prompt
        negative_prompt: Negative prompt
        steps: Number of inference steps (50 for high quality)
        guidance_scale: CFG scale (3.0 recommended for Playground)
        seed: Random seed for reproducibility
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=1024,
        width=1024
    ).images[0]
    
    return image

# ============================================================================
# MAIN HANDLER
# ============================================================================

def handler(event):
    """
    Main RunPod handler
    
    Expected input formats:
    
    1. Generate + Upscale:
    {
        "mode": "generate",
        "prompt": "beautiful landscape",
        "negative_prompt": "blurry, low quality",
        "steps": 50,
        "guidance_scale": 3.0,
        "seed": 12345,
        "upscale": true,
        "upscale_model": "realesrgan_x4",
        "target_dpi": 300
    }
    
    2. Upscale Only (from base64):
    {
        "mode": "upscale",
        "image_base64": "base64_string_here",
        "upscale_model": "realesrgan_anime",
        "target_dpi": 300
    }
    
    3. Upscale Only (from URL):
    {
        "mode": "upscale",
        "image_url": "https://...",
        "upscale_model": "realesrgan_x4",
        "target_dpi": 300
    }
    """
    
    try:
        input_data = event.get("input", {})
        mode = input_data.get("mode", "generate")
        
        # ===== MODE: GENERATE + OPTIONAL UPSCALE =====
        if mode == "generate":
            print("🎨 Generating image...")
            
            prompt = input_data.get("prompt")
            if not prompt:
                return {"error": "Prompt is required for generate mode"}
            
            image = generate_image(
                prompt=prompt,
                negative_prompt=input_data.get("negative_prompt", ""),
                steps=input_data.get("steps", 50),
                guidance_scale=input_data.get("guidance_scale", 3.0),
                seed=input_data.get("seed")
            )
            
            # Should we upscale?
            if input_data.get("upscale", False):
                print("🔼 Upscaling generated image...")
                image = upscale_image(
                    image,
                    target_dpi=input_data.get("target_dpi", 300),
                    upscale_model=input_data.get("upscale_model", "realesrgan_x4")
                )
            
            return {
                "status": "success",
                "image_base64": encode_image_to_base64(image),
                "width": image.width,
                "height": image.height,
                "mode": "generate_with_upscale" if input_data.get("upscale") else "generate_only"
            }
        
        # ===== MODE: UPSCALE ONLY =====
        elif mode == "upscale":
            print("🔼 Upscaling existing image...")
            
            # Get image from either base64 or URL
            if "image_base64" in input_data:
                image = decode_base64_image(input_data["image_base64"])
            elif "image_url" in input_data:
                image = download_image_from_url(input_data["image_url"])
            else:
                return {"error": "Either image_base64 or image_url is required for upscale mode"}
            
            upscaled = upscale_image(
                image,
                target_dpi=input_data.get("target_dpi", 300),
                upscale_model=input_data.get("upscale_model", "realesrgan_x4")
            )
            
            return {
                "status": "success",
                "image_base64": encode_image_to_base64(upscaled),
                "width": upscaled.width,
                "height": upscaled.height,
                "mode": "upscale_only"
            }
        
        else:
            return {"error": f"Invalid mode: {mode}. Use 'generate' or 'upscale'"}
    
    except Exception as e:
        return {"error": str(e)}

# Start the serverless function
runpod.serverless.start({"handler": handler})
