import os
import sys
import base64
import io
import traceback
import time

# Print Python and system info immediately
print("="*60)
print("STARTUP DIAGNOSTICS")
print("="*60)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Contents of /workspace: {os.listdir('/workspace') if os.path.exists('/workspace') else 'NOT FOUND'}")
print("="*60)

try:
    print("Importing torch...")
    import torch
    print(f"✓ torch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ FAILED to import torch: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("Importing torch.nn...")
    import torch.nn as nn
    print("✓ torch.nn imported")
except Exception as e:
    print(f"✗ FAILED to import torch.nn: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("Importing PIL...")
    from PIL import Image
    print("✓ PIL imported")
except Exception as e:
    print(f"✗ FAILED to import PIL: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("Importing runpod...")
    import runpod
    print(f"✓ runpod {runpod.__version__}")
except Exception as e:
    print(f"✗ FAILED to import runpod: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("Importing huggingface_hub...")
    import huggingface_hub
    print(f"✓ huggingface_hub {huggingface_hub.__version__}")
    
    # Backward-compatibility shim
    if not hasattr(huggingface_hub, "cached_download"):
        from huggingface_hub import file_download
        def _cached_download(*args, **kwargs):
            return file_download.hf_hub_download(*args, **kwargs)
        huggingface_hub.cached_download = _cached_download
        print("  Applied cached_download shim")
except Exception as e:
    print(f"✗ FAILED to import huggingface_hub: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("Importing diffusers...")
    import diffusers
    print(f"✓ diffusers {diffusers.__version__}")
    
    print("Importing StableDiffusionXLPipeline...")
    from diffusers import StableDiffusionXLPipeline
    print("✓ StableDiffusionXLPipeline imported")
    
    print("Importing EDMDPMSolverMultistepScheduler...")
    from diffusers import EDMDPMSolverMultistepScheduler
    print("✓ EDMDPMSolverMultistepScheduler imported")
except Exception as e:
    print(f"✗ FAILED to import diffusers components: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("Importing transformers...")
    from transformers import AutoModel, AutoProcessor
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ FAILED to import transformers: {e}")
    traceback.print_exc()
    sys.exit(1)

print("="*60)
print("ALL IMPORTS SUCCESSFUL")
print("="*60)

# Add GitHub repo to path
AESTHETIC_REPO_PATH = os.getenv("AESTHETIC_REPO_PATH", "/workspace/aesthetic-predictor-v2-5")
sys.path.append(AESTHETIC_REPO_PATH)

# Paths verified on your volume (override with env vars when needed)
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/playground-v2.5-1024px-aesthetic.fp16.safetensors")
SIGLIP_PATH = os.getenv("SIGLIP_PATH", "/workspace/siglip")
SIGLIP_MODEL_FILE = os.getenv("SIGLIP_MODEL_FILE", "model.safetensors")
PREDICTOR_WEIGHTS = os.getenv(
    "PREDICTOR_WEIGHTS",
    os.path.join(AESTHETIC_REPO_PATH, "models", "aesthetic_predictor_v2_5.pth")
)
WAIT_FOR_ASSETS_SECONDS = int(os.getenv("WAIT_FOR_ASSETS_SECONDS", "90"))

print("\nChecking file paths...")
print(f"MODEL_PATH exists: {os.path.exists(MODEL_PATH)}")
print(f"SIGLIP_PATH exists: {os.path.exists(SIGLIP_PATH)}")
print(f"PREDICTOR_WEIGHTS exists: {os.path.exists(PREDICTOR_WEIGHTS)}")
print(f"aesthetic-predictor-v2-5 dir exists: {os.path.exists(AESTHETIC_REPO_PATH)}")

pipe = None
aesthetic_model = None
aesthetic_processor = None

def load_models():
    global pipe, aesthetic_model, aesthetic_processor

    def _looks_local(path: str) -> bool:
        return path.startswith("/") or path.startswith(".") or os.path.splitdrive(path)[0] != ""

    def _wait_for_path(path, is_dir=False):
        """Give network volumes time to mount before failing."""
        if WAIT_FOR_ASSETS_SECONDS <= 0:
            return False

        deadline = time.time() + WAIT_FOR_ASSETS_SECONDS
        check_fn = os.path.isdir if is_dir else os.path.isfile
        while time.time() < deadline:
            if check_fn(path):
                return True
            time.sleep(1)
        return False

    def _require_path(path, description, env_var=None, is_dir=False):
        if _looks_local(path):
            exists = os.path.isdir(path) if is_dir else os.path.isfile(path)
            if not exists:
                if _wait_for_path(path, is_dir=is_dir):
                    exists = True
                if not exists:
                    location_hint = (
                        f"Set {env_var} to a valid {'directory' if is_dir else 'file'} path "
                        f"or ensure the network volume is mounted at {path}."
                    ) if env_var else f"Ensure the network volume provides the asset at {path}."
                raise FileNotFoundError(
                    f"{description} not found at {path}. {location_hint}"
                )
        else:
            print(f"{description} appears remote ({path}); skipping local existence check.")
        return True
    
    if pipe is None:
        print(f"\nLoading Playground from {MODEL_PATH}")
        _require_path(MODEL_PATH, "Playground checkpoint", env_var="MODEL_PATH")
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
            
            print("✓ Pipeline loaded successfully")
            print(f"  Scheduler: {type(pipe.scheduler).__name__}")
        except Exception as e:
            print(f"✗ Error loading pipeline: {e}")
            traceback.print_exc()
            raise

    if aesthetic_model is None:
        print(f"\nLoading Predictor from {SIGLIP_PATH}")
        _require_path(SIGLIP_PATH, "SigLIP model directory", env_var="SIGLIP_PATH", is_dir=True)
        _require_path(PREDICTOR_WEIGHTS, "Aesthetic predictor weights", env_var="PREDICTOR_WEIGHTS")
        try:
            from aesthetic_predictor_v2_5 import AestheticPredictorV2_5
            
            backbone = AutoModel.from_pretrained(SIGLIP_PATH).to("cuda")
            aesthetic_processor = AutoProcessor.from_pretrained(SIGLIP_PATH)
            
            aesthetic_model = AestheticPredictorV2_5(backbone).to("cuda")
            aesthetic_model.load_state_dict(torch.load(PREDICTOR_WEIGHTS, map_location="cuda"))
            aesthetic_model.eval()
            print("✓ Aesthetic model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading aesthetic model: {e}")
            traceback.print_exc()
            raise

def handler(job):
    print("\n" + "="*60)
    print("HANDLER CALLED")
    print("="*60)
    try:
        load_models()
        inp = job.get("input", {})
        route = inp.get("route", "create")
        prompt = inp.get("prompt", "A high quality photo")
        seed = inp.get("seed", None)
        
        # Route-specific settings
        if route == "create":
            num_steps = 25
            guidance_scale = 3.0
        elif route == "production":
            num_steps = 50
            guidance_scale = 3.0
        else:
            num_steps = inp.get("num_inference_steps", 25)
            guidance_scale = inp.get("guidance_scale", 3.0)
        
        # Set seed if provided, otherwise generate one
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        print(f"Route: {route}")
        print(f"Prompt: {prompt}")
        print(f"Steps: {num_steps}, Guidance: {guidance_scale}, Seed: {seed}")
        
        # Generation logic
        print("Starting image generation...")
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        print("✓ Image generated, calculating aesthetic score...")
        
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
        
        print(f"✓ SUCCESS! Aesthetic score: {result['aesthetic_score']}")
        return result
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"✗ HANDLER ERROR:\n{error_msg}")
        return {"error": error_msg}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING RUNPOD HANDLER")
    print("="*60)
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"✗ FATAL ERROR starting handler: {e}")
        traceback.print_exc()
        sys.exit(1)
