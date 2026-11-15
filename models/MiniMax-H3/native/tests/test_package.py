"""checkpoints.lock.json: config.json first, sizes add up, paths stay inside, both presets complete; prepare() picks
the shared files plus one preset and refuses to download from an unpinned revision."""
import json
from pathlib import Path
import sys

import pytest

NATIVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NATIVE))
from paiton_h3v import weights  # noqa: E402

LOCK = json.loads((NATIVE / "checkpoints.lock.json").read_text())


def test_lock():
    files = LOCK["files"]
    assert files[0]["file"] == "config.json"
    assert LOCK["download_bytes"] == sum(r["bytes"] for r in files)
    assert len({r["file"] for r in files}) == len(files)
    for r in files:
        assert not r["file"].startswith("/") and ".." not in r["file"].split("/") and len(r["sha256"]) == 64
    for preset in ("turbo4", "turbo8"):
        names = {r["file"] for r in files if r.get("preset") == preset}
        assert names == {f"weights/{preset}/{n}" for n in ("dit.bin", "dit.idx", "vae_latents.bin")}
    shared = {r["file"].split("/")[0] for r in files if "preset" not in r}
    assert shared == {"config.json", "te-w8", "dec8.bin", "enc.bin", "audio.bin", "model"}


def test_prepare_preset_and_unpinned(tmp_path):
    with pytest.raises(ValueError, match="unknown preset"):
        weights.prepare(tmp_path, "base50")
    if LOCK["revision"] == "unpublished":
        with pytest.raises(ValueError, match="not pinned"):
            weights.prepare(tmp_path, "turbo4")
