"""
Test script to verify all imports work
Run this to debug import issues
"""

import sys
import huggingface_hub

# Backward-compatibility shim for diffusers<=0.25 expecting cached_download
# huggingface_hub 0.23+ removed cached_download; remap to hf_hub_download when missing
if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import file_download

    def _cached_download(*args, **kwargs):
        return file_download.hf_hub_download(*args, **kwargs)

    huggingface_hub.cached_download = _cached_download

import diffusers

print("Python version:", sys.version)
print("\nTesting imports...")
print("=" * 60)

try:
    import torch
    print(f"✓ torch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ torch: {e}")
    sys.exit(1)

try:
    import diffusers
    print(f"✓ diffusers {diffusers.__version__}")
except Exception as e:
    print(f"✗ diffusers: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ transformers: {e}")
    sys.exit(1)

try:
    import accelerate
    print(f"✓ accelerate {accelerate.__version__}")
except Exception as e:
    print(f"✗ accelerate: {e}")
    sys.exit(1)

try:
    import runpod
    print(f"✓ runpod {runpod.__version__}")
except Exception as e:
    print(f"✗ runpod: {e}")
    sys.exit(1)

try:
    from PIL import Image
    import PIL
    print(f"✓ PIL {PIL.__version__}")
except Exception as e:
    print(f"✗ PIL: {e}")
    sys.exit(1)

try:
    import huggingface_hub
    print(f"✓ huggingface_hub {huggingface_hub.__version__}")
except Exception as e:
    print(f"✗ huggingface_hub: {e}")
    sys.exit(1)

print("=" * 60)
print("All imports successful!")
print("\nTesting specific classes...")
print("=" * 60)

try:
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    EDMDPMSolverMultistepScheduler = getattr(diffusers, "EDMDPMSolverMultistepScheduler", DPMSolverMultistepScheduler)
    print("✓ DiffusionPipeline and schedulers (EDM fallback-safe)")
except Exception as e:
    print(f"✗ DiffusionPipeline: {e}")
    sys.exit(1)

try:
    from transformers import AutoModel, AutoProcessor
    print("✓ AutoModel and AutoProcessor")
except Exception as e:
    print(f"✗ AutoModel/AutoProcessor: {e}")
    sys.exit(1)

print("=" * 60)
print("All tests passed!")
