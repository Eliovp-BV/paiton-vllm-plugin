"""Download and verify the published checkpoint; never execute Hub code."""

import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent.parent


MODEL_LOCKS = {"original": "checkpoint.lock.json", "uncensored": "checkpoint.uncensored.lock.json"}


def checkpoint_lock(model="original"):
    if model not in MODEL_LOCKS:
        raise ValueError(f"Unknown model variant: {model}")
    return json.loads((PACKAGE / MODEL_LOCKS[model]).read_text())


def verify(directory, lock=None):
    directory = Path(directory)
    lock = checkpoint_lock() if lock is None else lock
    for row in lock["files"]:
        relative = Path(row["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid checkpoint manifest path")
        path = directory / relative
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Missing or incomplete checkpoint file: {relative}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != row["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {relative}")
    return directory.resolve()


def prepare(model_dir=None, cache_dir=None, offline=False, model="original"):
    lock = checkpoint_lock(model)
    if model_dir is None:
        revision = lock.get("revision", "")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("This model's publication revision has not been pinned yet")
        from huggingface_hub import snapshot_download

        print(f"Preparing {lock['repository']} at {lock['revision']} "
              f"({lock['download_bytes'] / 1e9:.2f} GB on first use)...",
              file=sys.stderr, flush=True)
        model_dir = snapshot_download(
            repo_id=lock["repository"], revision=lock["revision"],
            allow_patterns=[row["file"] for row in lock["files"]],
            cache_dir=cache_dir, local_files_only=offline, max_workers=2,
        )
    print("Verifying checkpoint SHA-256 hashes...", file=sys.stderr, flush=True)
    return verify(model_dir, lock)
