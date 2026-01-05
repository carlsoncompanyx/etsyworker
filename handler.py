import os
import sys
import base64
import io
import traceback
import time
import subprocess

# ============================================================
# STARTUP DIAGNOSTICS
# ============================================================
print("=" * 60)
print("STARTUP DIAGNOSTICS")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Contents of /workspace: {os.listdir('/workspace') if os.path.exists('/workspace') else 'NOT FOUND'}")
print("=" * 60)

# ============================================================
# IMPORTS (fail-fast with clear logs)
# ============================================================
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

print("=" * 60)
print("ALL IMPORTS SUCCESSFUL")
print("=" * 60)

# ============================================================
# VOLUME ROOT DETECTION (fixes serverless mount mismatches)
# ============================================================
CHECKPOINT_FILENAME = "playground-v2.5-1024px-aesthetic.fp16.safetensors"

def _safe_ls(path: str, limit: int = 200) -> str:
    try:
        out = subprocess.getoutput(f"ls -la {path}")
        lines = out.splitlines()
        if len(lines) > limit:
            return "\n".join(lines[:limit] + [f"... ({len(lines) - limit} more lines)"])
        return out
    except Exception as e:
        return f"ls failed: {e}"

def detect_volume_root(default: str = "/workspace") -> str:
    """
    Detect where the network volume is mounted in THIS runtime.
    We prioritize env VOLUME_ROOT if set, then try common mount points,
    then attempt a bounded search.
    """
    env_root = os.getenv("VOLUME_ROOT")
    candidates = [env_root, default, "/runpod-volume", "/network-volume", "/volume", "/data", "/mnt"]
    candidates = [c for c in candidates if c]

    # 1) Direct file check at candidate roots
    for root in candidates:
        p = os.path.join(root, CHECKPOINT_FILENAME)
        if os.path.isfile(p):
            return root

    # 2) Check one level down (some templates mount into a subdir)
    for root in candidates:
        if os.path.isdir(root):
            try:
                for child in os.listdir(root):
                    p = os.path.join(root, child, CHECKPOINT_FILENAME)
                    if os.path.isfile(p):
                        return os.path.join(root, child)
            except Exception:
                pass

    # 3) Bounded walk starting at / (kept shallow to avoid runaway)
    try:
        for dirpath, dirnames, filenames in os.walk("/"):
            if CHECKPOINT_FILENAME in filenames:
                return dirpath
            # Keep it shallow: do not descend too deep
            depth = dirpath.count(os.sep)
            if depth > 4:
                dirnames[:] = []
    except Exception:
        pass

    return default

VOLUME_ROOT = detect_volume_root("/workspace")
print(f"\nUsing VOLUME_ROOT={VOLUME_ROOT}")

# ============================================================
# PATHS (built from VOLUME_ROOT; env vars can still override)
# ============================================================
AESTHETIC_REPO_PATH = os.getenv("AESTHETIC_REPO_PATH", os.path.join(VOLUME_ROOT, "aesthetic-predictor-v2-5"))
sys.path.append(AESTHETIC_REPO_PATH)

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(VOLUME_ROOT, CHECKPOINT_FILENAME))
SIGLIP_PATH = os.getenv("SIGLIP_PATH", os.path.join(VOLUME_ROOT, "siglip"))
SIGLIP_MODEL_FILE = os.getenv("SIGLIP_MODEL_FILE", "model.safetensors")
PREDICTOR_WEIGHTS = os.getenv(
    "PREDICTOR_WEIGHTS",
    os.path.join(AESTHETIC_REPO_PATH, "models", "aesthetic_predictor_v2_5.pth")
)

# Increase default wait to be more tolerant of slow mounts
WAIT_FOR_ASSETS_SECONDS = int(os.getenv("WAIT_FOR_ASSETS_SECONDS", "180"))

print("\n" + "=" * 60)
print("ENV + FILESYSTEM DIAGNOSTICS")
print("=" * 60)
for k in [
    "VOLUME_ROOT", "MODEL_PATH", "SIGLIP_PATH", "SIGLIP_MODEL_FILE",
    "AESTHETIC_REPO_PATH", "PREDICTOR_WEIGHTS", "WAIT_FOR_ASSETS_SECONDS",
    "HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"
]:
    print(f"{k}={os.getenv(k)}")

print("\n--- df -h ---")
print(subprocess.getoutput("df -h | head -n 80"))

print("\n--- mount (filtered) ---")
print(subprocess.getoutput("mount | grep -E '/workspace|runpod|volume|network' || true"))

print("\n--- ls -la /workspace ---")
print(_safe_ls("/workspace"))

print("\n--- ls -la VOLUME_ROOT ---")
print(_safe_ls(VOLUME_ROOT))

print("=" * 60)

print("\nChecking file paths (type-aware)...")
print(f"MODEL_PATH isfile: {os.path.isfile(MODEL_PATH)}  -> {MODEL_PATH}")
print(f"SIGLIP_PATH isdir: {os.path.isdir(SIGLIP_PATH)}  -> {SIGLIP_PATH}")
print(f"PREDICTOR_WEIGHTS isfile: {os.path.isfile(PREDICTOR_WEIGHTS)}  -> {PREDICTOR_WEIGHTS}")
print(f"AESTHETIC_REPO_PATH isdir: {os.path.isdir(AESTHETIC_REPO_PATH)}  -> {AESTHETIC_REPO_PATH}")

pipe = None
aesthetic_model = None
aesthetic_processor = None

# ============================================================
# MODEL LOADING
# ============================================================
def load_models():
    global pipe, aesthetic_model, aesthetic_processor

    def _looks_local(path: str) -> bool:
        return path.startswith("/") or path.startswith(".") or os.path.splitdrive(path)[0] != ""

    def _wait_for_path(path: str, is_dir: bool = False) -> bool:
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

    def _require_path(path: str, description: str, env_var: str = None, is_dir: bool = False) -> bool:
        if _looks_local(path):
            exists = os.path.isdir(path) if is_dir else os.path.isfile(path)
            if not exists:
                # DEBUG: prove mount/visibility at failure time
                print("\n" + "=" * 60)
                print("ASSET MISSING DEBUG")
                print("=" * 60)
                print(f"description: {description}")
                print(f"expected path: {path}")
                print(f"is_dir expected: {is_dir}")
                print(f"cwd: {os.getcwd()}")
                print("--- df -h ---")
                print(subprocess.getoutput("df -h | head -n 80"))
                print("--- mount (filtered) ---")
                print(subprocess.getoutput("mount | grep -E '/workspace|runpod|volume|network' || true"))
                print("--- ls -la /workspace ---")
                print(_safe_ls("/workspace"))
                print(f"--- ls -la {VOLUME_ROOT} ---")
                print(_safe_ls(VOLUME_ROOT))
                print("--- ls -la / ---")
                print(subprocess.getoutput("ls -la / | head -n 200"))
                print("=" * 60 + "\n")

                # Wait once more (in case mount is late)
                if _wait_for_path(path, is_dir=is_dir):
                    exists = True

                if not exists:
                    location_hint = (
                        f"Set {env_var} to a valid {'directory' if is_dir else 'file'} path "
                        f"or ensure the network volume is mounted to include {path}."
                    ) if env_var else f"Ensure the network volume provides the asset at {path}."

                    raise FileNotFoundError(
                        f"{description} not found at {path}. {location_hint}"
                    )
        else:
            print(f"{description} appears remote ({path}); skipping local existence check.")
        return True

    if pipe is None:
        print(f"\nLoading Playground from {MODEL_PATH}")
        _require_path(MODEL_PATH, "Playground checkpoint", env_var="MODEL_PATH", is_dir=False)
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                MODEL_PATH,
                torch_dtype=torch.float16,
                use_safetensors=True,
                load_safety_checker=False
            ).to("cuda")

            pipe.scheduler = EDMDPMSolverMultistepScheduler()
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

        # Optional: validate the expected weights file inside SigLIP dir if you want
        # _require_path(os.path.join(SIGLIP_PATH, SIGLIP_MODEL_FILE), "SigLIP model file", env_var="SIGLIP_MODEL_FILE")

        _require_path(PREDICTOR_WEIGHTS, "Aesthetic predictor weights", env_var="PREDICTOR_WEIGHTS", is_dir=False)

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

# ============================================================
# HANDLER
# ============================================================
def handler(job):
    print("\n" + "=" * 60)
    print("HANDLER CALLED")
    print("=" * 60)
    try:
        load_models()

        inp = job.get("input", {})
        route = inp.get("route", "create")
        prompt = inp.get("prompt", "A high quality photo")
        seed = inp.get("seed", None)

        # Route-specific defaults
        if route == "create":
            num_steps = 25
            guidance_scale = 3.0
        elif route == "production":
            num_steps = 50
            guidance_scale = 3.0
        else:
            num_steps = inp.get("num_inference_steps", 25)
            guidance_scale = inp.get("guidance_scale", 3.0)

        # Seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"Route: {route}")
        print(f"Prompt: {prompt}")
        print(f"Steps: {num_steps}, Guidance: {guidance_scale}, Seed: {seed}")

        # Generate
        print("Starting image generation...")
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        print("✓ Image generated, calculating aesthetic score...")

        # Score
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
            "seed": seed,
            "volume_root": VOLUME_ROOT,
        }

        print(f"✓ SUCCESS! Aesthetic score: {result['aesthetic_score']}")
        return result

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"✗ HANDLER ERROR:\n{error_msg}")
        return {"error": error_msg}

# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING RUNPOD HANDLER")
    print("=" * 60)
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"✗ FATAL ERROR starting handler: {e}")
        traceback.print_exc()
        sys.exit(1)
