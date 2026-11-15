"""Image build check: every file in artifacts/manifest.json is present with its SHA-256, and nothing else ships."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/app")
manifest = json.loads((root / "manifest.json").read_text())
shipped = {str(p.relative_to(root)) for d in ("lib",) for p in (root / d).rglob("*") if p.is_file()}
assert shipped == set(manifest["files"]), f"unlisted or missing artifacts: {sorted(shipped ^ set(manifest['files']))}"
for name, sha in manifest["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, f"SHA-256 mismatch: {name}"
dirty = [k for k, v in manifest["components"].items() if v["dirty_files"]]
print(f"artifacts OK ({len(shipped)} files); components with uncommitted changes: {dirty or 'none'}")
