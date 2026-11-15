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


def test_text_encoder_from_image_repo(tmp_path, monkeypatch):
    te = [r for r in LOCK["files"] if r["file"].startswith("te-w8/")]
    assert len(te) == 52
    for r in te:
        assert r["repository"] == "EliovpAI/MiniMax-H3-Image-FP8-Paiton-RDNA4" and len(r["revision"]) == 40
        assert r["source"] == "te/" + r["file"].split("/", 1)[1]
    calls = []

    def fake(repo, revision, name, path, token):
        calls.append((repo, revision, name))
        raise RuntimeError("stop")

    lock = dict(LOCK, revision="0" * 40, files=[LOCK["files"][0], te[0]])
    monkeypatch.setattr(weights, "LOCK", tmp_path / "lock.json")
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    (tmp_path / "c" / "MiniMax-H3-FP8-Paiton-RDNA4").mkdir(parents=True)
    (tmp_path / "c" / "MiniMax-H3-FP8-Paiton-RDNA4" / "config.json").write_bytes(b"x" * LOCK["files"][0]["bytes"])
    monkeypatch.setattr(weights, "_sha256", lambda p: LOCK["files"][0]["sha256"])
    monkeypatch.setattr(weights, "_download", fake)
    lock["files"].append({"file": "weights/turbo4/dit.bin", "bytes": 1, "sha256": "0" * 64, "preset": "turbo4"})
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    with pytest.raises(RuntimeError, match="stop"):
        weights.prepare(tmp_path / "c", "turbo4")
    assert calls == [(te[0]["repository"], te[0]["revision"], te[0]["source"])]
