FROM runpod/pytorch:2.3.0-py3.10-cuda12.1

ARG UPSCAYL_VERSION=latest
ARG GITHUB_TOKEN
ENV UPSCAYL_VERSION=${UPSCAYL_VERSION} \
    OUTPUT_DIR=/app/output

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tar \
        xz-utils \
        libgtk-3-0 \
        libnss3 \
        libasound2 \
        libatk-bridge2.0-0 \
        libx11-xcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libxkbcommon0 \
        libpango-1.0-0 \
        libgbm1 \
        libatk1.0-0 \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    mkdir -p /opt/upscayl /app/output; \
    GITHUB_TOKEN=${GITHUB_TOKEN:-} python - "$UPSCAYL_VERSION" <<'PY'
import json
import os
import shutil
import sys
import pathlib
import tarfile
import tempfile
import urllib.request
import urllib.error
import zipfile

USER_AGENT = "runpod-upscaler-build"
API_ROOT = "https://api.github.com/repos/upscayl/upscayl/releases"

TOKEN = os.environ.get("GITHUB_TOKEN")


class DownloadError(RuntimeError):
    pass

def resolve_release(version: str) -> dict:
    if version == "latest":
        endpoint = f"{API_ROOT}/latest"
    else:
        tag = version.split("/", 1)[-1]
        endpoint = f"{API_ROOT}/tags/{tag}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    request = urllib.request.Request(endpoint, headers=headers)
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def download_asset(url: str, destination: str) -> None:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/octet-stream",
    }
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response, open(destination, "wb") as target:
            shutil.copyfileobj(response, target)
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        raise DownloadError(f"Failed to download Upscayl asset: {exc}") from exc


def safe_extract(archive: tarfile.TarFile, destination: str) -> None:
    destination_path = pathlib.Path(destination).resolve()
    for member in archive.getmembers():
        member_path = destination_path.joinpath(member.name).resolve()
        if destination_path not in member_path.parents and destination_path != member_path:
            raise RuntimeError("Archive member escapes extraction directory")
    archive.extractall(path=destination)


def extract_cli(archive_path: str, install_dir: str) -> str:
    if os.path.exists(install_dir):
        shutil.rmtree(install_dir)
    os.makedirs(install_dir, exist_ok=True)
    archive_lower = archive_path.lower()
    if archive_lower.endswith(('.tar.xz', '.tar.gz', '.tgz', '.tar')):
        with tarfile.open(archive_path, mode="r:*") as archive:
            safe_extract(archive, install_dir)
    elif archive_lower.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(path=install_dir)
    else:
        raise RuntimeError(f"Unsupported Upscayl archive format: {archive_path}")
    for root, _dirs, files in os.walk(install_dir):
        for candidate in ("upscayl-cli", "upscayl"):
            if candidate in files:
                return os.path.join(root, candidate)
    raise RuntimeError("upscayl CLI binary not found in archive")


def main() -> None:
    version = sys.argv[1]
    release = resolve_release(version)
    def score_asset(item: dict) -> int:
        name = item.get("name", "").lower()
        score = 0
        if "cli" in name:
            score += 4
        if "linux" in name:
            score += 2
        if name.endswith(".tar.xz"):
            score += 3
        elif name.endswith((".tar.gz", ".tgz")):
            score += 2
        elif name.endswith(".zip"):
            score += 1
        return score

    assets = release.get("assets", [])
    ranked = sorted(
        (item for item in assets if score_asset(item) > 0),
        key=score_asset,
        reverse=True,
    )
    asset = ranked[0] if ranked else None
    if asset is None:
        raise RuntimeError("Upscayl CLI Linux asset not found in release")

    with tempfile.TemporaryDirectory() as workdir:
        asset_name = os.path.basename(asset.get("name", "upscayl-cli.tar.xz"))
        archive_path = os.path.join(workdir, asset_name)
        download_asset(asset["browser_download_url"], archive_path)
        binary_path = extract_cli(archive_path, "/opt/upscayl")

    destination = "/usr/local/bin/upscayl-cli"
    if os.path.islink(destination) or os.path.isfile(destination):
        os.unlink(destination)
    os.symlink(binary_path, destination)
    os.chmod(binary_path, 0o755)


if __name__ == "__main__":
    main()
PY

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "server.py"]
