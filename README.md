# RunPod Serverless: Playground v2.5 + Aesthetic Scoring

This endpoint is optimized to use a persistent network volume for instant cold starts.

## Volume Setup
Ensure your RunPod Network Volume is mounted at `/workspace` with the following files:
- `/workspace/playground-v2.5-1024px-aesthetic.fp16.safetensors`
- `/workspace/siglip/` (Full SigLIP model directory)
- `/workspace/aesthetic-predictor-v2-5/` (Cloned GitHub repo containing weights)

## API Routes

### 1. `/create` (Fast)
Generates an image and returns an aesthetic score (1-10).
- **Input**: `{"route": "create", "prompt": "..."}`
- **Estimated Time**: 10-15s

### 2. `/production` (High Quality)
Generates a high-fidelity image using a fixed seed from the create route.
- **Input**: `{"route": "production", "prompt": "...", "seed": 12345}`
- **Estimated Time**: 25-40s

## Deployment
1. Build and push your Docker image.
2. Create a RunPod Serverless Template.
3. **Crucial**: Link your Network Volume to the template at the `/workspace` mount point.
