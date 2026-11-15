"""checkpoints.lock.json: sizes add up, paths stay inside, every file the runner opens is listed."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCK = json.loads((ROOT / "checkpoints.lock.json").read_text())


def test_lock():
    files = LOCK["files"]
    assert LOCK["download_bytes"] == sum(r["bytes"] for r in files)
    assert len({r["file"] for r in files}) == len(files)
    for r in files:
        assert not r["file"].startswith("/") and ".." not in r["file"].split("/") and len(r["sha256"]) == 64
    names = {r["file"] for r in files}
    from paiton_ltx25.__main__ import runner_command   # noqa: E402 (needs tokenizers, as in the image)
    cmd = runner_command(Path("/lib"), Path("/b"))
    opened = [v[3:] for v in cmd if v.startswith("/b/") and not v.endswith(("/dit", "/te"))]
    assert opened and all(f in names for f in opened), opened
    for d in ("dit/dit8.bin", "dit/dit8.idx", "dit/dit.idx", "dit/tables.bin", "dit/tables.idx", "te/tokenizer.json"):
        assert d in names


if __name__ == "__main__":
    test_lock()
    print("ok")
