"""First-run weight preparation: download the pinned Paiton FP8 bundle of MiniMax-H3-Image and verify every SHA-256."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

APP = Path(__file__).resolve().parent.parent
LOCK = APP / "checkpoints.lock.json"
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


def prepare(cache):
    """Download and verify every locked file into cache/<bundle>/ and return that directory. A verified file is
    recorded and not hashed again on later starts."""
    lock = json.loads(LOCK.read_text())
    bundle = cache / lock["repository"].split("/")[-1]
    token = os.environ.get("HF_TOKEN")
    pinned = len(lock["revision"]) == 40 and all(c in "0123456789abcdef" for c in lock["revision"])
    log(f"Checking {lock['repository']} at {lock['revision']} ({lock['download_bytes'] / 1e9:.1f} GB on first use)")
    for row in lock["files"]:
        rel = Path(row["file"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"invalid path in the checkpoint lock: {rel}")
        path, mark = bundle / rel, bundle / ".verified" / (row["file"].replace("/", "__") + ".sha256")
        if path.is_file() and mark.is_file() and mark.read_text() == row["sha256"] and path.stat().st_size == row["bytes"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if not (path.is_file() and path.stat().st_size == row["bytes"]):
            if not pinned:
                raise ValueError(f"{rel} is missing and the weights repository revision is not pinned yet")
            log(f"Downloading {rel} ({row['bytes'] / 1e9:.2f} GB)")
            _download(lock["repository"], lock["revision"], row["file"], path, token)
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            path.unlink()
            raise ValueError(f"SHA-256 or size mismatch for {rel}; the file was removed, start again to download it")
        mark.parent.mkdir(exist_ok=True)
        mark.write_text(row["sha256"])
    return bundle
