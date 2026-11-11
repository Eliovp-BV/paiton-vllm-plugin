"""First-run weight preparation: download the pinned MiniMax-H3-Image files, verify SHA-256, convert to the native bundle."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

APP = Path(__file__).resolve().parent.parent
LOCK = APP / "checkpoints.lock.json"
CONVERT = APP / "convert"
CHUNK = 16 << 20


def log(message):
    print(message, file=sys.stderr, flush=True)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _download(repo, revision, name, path, token):
    url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')}/{repo}/resolve/{revision}/{name}"
    part = path.with_name(path.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "paiton-minimax-h3-image"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if have:
        headers["Range"] = f"bytes={have}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        if have and response.status != 206:   # server ignored the range: start over
            have = 0
        with part.open("ab" if have else "wb") as out:
            shutil.copyfileobj(response, out, CHUNK)
    part.rename(path)


def fetch(source):
    """Download and verify every locked file into source/. A verified file is recorded and not hashed again."""
    lock = json.loads(LOCK.read_text())
    token = os.environ.get("HF_TOKEN")
    log(f"Checking {lock['repository']} at {lock['revision']} ({lock['download_bytes'] / 1e9:.0f} GB on first use)")
    for row in lock["files"]:
        rel = Path(row["file"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"invalid path in the checkpoint lock: {rel}")
        path, mark = source / rel, source / ".verified" / (row["file"].replace("/", "__") + ".sha256")
        if path.is_file() and mark.is_file() and mark.read_text() == row["sha256"] and path.stat().st_size == row["bytes"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if not (path.is_file() and path.stat().st_size == row["bytes"]):
            log(f"Downloading {rel} ({row['bytes'] / 1e9:.2f} GB)")
            _download(lock["repository"], lock["revision"], row["file"], path, token)
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            path.unlink()
            raise ValueError(f"SHA-256 or size mismatch for {rel}; the file was removed, start again to download it")
        mark.parent.mkdir(exist_ok=True)
        mark.write_text(row["sha256"])
    return lock


def prepare(cache):
    """Return the converted bundle directory, building it on the first run."""
    source, bundle = cache / "minimax-h3-image", cache / "native-bundle"
    lock = fetch(source)
    stamp = {"revision": lock["revision"], "converters": json.loads((APP / "manifest.json").read_text())["converters"]}
    if (bundle / "READY.json").is_file() and json.loads((bundle / "READY.json").read_text()) == stamp:
        return bundle
    work = cache / "native-bundle.tmp"
    shutil.rmtree(work, ignore_errors=True)
    (work / "dit").mkdir(parents=True)
    py = [sys.executable, "-s"]   # sys.path[0] is /app/convert (te_convert imports real_weights from there), never the cache
    steps = [
        ("DiT (FP8)", py + [CONVERT / "dit_convert.py", "--model", source, "--out", work / "dit"]),
        ("text encoder (FP8)", py + [CONVERT / "te_convert.py", "--model", source / "text_encoder", "--out", work / "te", "--format", "w8"]),
        ("VAE decoder (int8)", py + [CONVERT / "vae_pack.py", "--int8", source / "vae/diffusion_pytorch_model.safetensors", work / "dec8.bin"]),
        ("VAE encoder", py + [CONVERT / "vae_pack.py", "--encoder", source / "vae/diffusion_pytorch_model.safetensors", work / "enc.bin"]),
    ]
    for label, command in steps:
        log(f"Converting the {label}; this runs once")
        subprocess.run([str(c) for c in command], check=True, stdout=sys.stderr)
    (work / "READY.json").write_text(json.dumps(stamp))
    shutil.rmtree(bundle, ignore_errors=True)
    work.rename(bundle)
    return bundle
