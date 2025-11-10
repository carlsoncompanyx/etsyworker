import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_DIR", "/app/output"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ultrasharp"
SCALE = "4"
FORMAT = "webp"
QUALITY = "100"


def _download_image(url: str, download_dir: Path) -> Path:
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    parsed = urlparse(url)
    extension = Path(parsed.path).suffix or ".png"
    filename = download_dir / f"source_{uuid.uuid4().hex}{extension}"
    filename.write_bytes(response.content)
    return filename


def _run_upscayl(input_path: Path, output_path: Path) -> Path:
    temp_output = output_path.parent / f"{output_path.stem}_tmp"
    temp_output.mkdir(parents=True, exist_ok=True)

    command: List[str] = [
        "upscayl-cli",
        "--input",
        str(input_path),
        "--output",
        str(temp_output),
        "--mode",
        MODEL_NAME,
        "--scale",
        SCALE,
        "--format",
        FORMAT,
        "--quality",
        QUALITY,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - logged for debugging
        raise RuntimeError(
            f"Upscayl CLI failed for {input_path.name}: {exc.stderr.decode('utf-8', errors='ignore')}"
        ) from exc

    generated_files = sorted(temp_output.glob(f"*.{FORMAT}"))
    if not generated_files:
        raise RuntimeError(f"Upscayl CLI did not produce any {FORMAT} files for {input_path.name}.")

    generated = generated_files[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(generated), output_path)
    shutil.rmtree(temp_output, ignore_errors=True)
    return output_path


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    input_payload = event.get("input") or {}
    image_urls = input_payload.get("image_urls")
    if not image_urls or not isinstance(image_urls, list):
        raise ValueError("'image_urls' must be a non-empty list of URLs.")

    outputs: List[str] = []
    temp_dir = Path("/tmp") / f"upscayl_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for url in image_urls:
            source_path = _download_image(url, temp_dir)
            output_name = f"upscaled_{uuid.uuid4().hex}.{FORMAT}"
            final_output_path = OUTPUT_ROOT / output_name
            upscaled_path = _run_upscayl(source_path, final_output_path)
            outputs.append(str(upscaled_path))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"outputs": outputs}
