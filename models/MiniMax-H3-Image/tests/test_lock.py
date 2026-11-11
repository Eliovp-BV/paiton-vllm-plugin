"""checkpoints.lock.json: config.json first (Hugging Face counts downloads on it), sizes add up, paths stay inside."""
import json
from pathlib import Path

LOCK = json.loads((Path(__file__).resolve().parents[1] / "checkpoints.lock.json").read_text())


def test_lock():
    files = LOCK["files"]
    assert files[0]["file"] == "config.json" and len(files) == 59
    assert LOCK["download_bytes"] == sum(r["bytes"] for r in files)
    assert len({r["file"] for r in files}) == len(files)
    for r in files:
        assert not r["file"].startswith("/") and ".." not in r["file"].split("/") and len(r["sha256"]) == 64
