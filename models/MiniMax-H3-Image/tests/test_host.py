"""Host preprocessing checks. The golden checks need H3I_GOLDENS (reference captures) and H3I_MODEL (the checkpoint)."""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paiton_h3i import host  # noqa: E402

GOLDENS, MODEL = os.environ.get("H3I_GOLDENS"), os.environ.get("H3I_MODEL")
needs_goldens = pytest.mark.skipif(not (GOLDENS and MODEL), reason="set H3I_GOLDENS and H3I_MODEL")


def test_canvas():
    assert host.canvas_size(1024, 1024) == (2048, 2048)
    assert host.canvas_size(4096, 4096) == (4096, 4096)
    assert host.canvas_size(3552, 4736) == (3552, 4736)
    assert host.bound_short_edge(1024, 1024, 1024, 2048) == (1024, 1024)
    assert host.bound_short_edge(3000, 4000, 1024, 1024) == (1024, 1376)


def _golden(path):
    """safetensors -> numpy; bf16 widened to f32 (numpy has no bf16)."""
    import json, struct
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    out = {}
    for k, v in json.loads(raw[8:8 + n]).items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        buf = raw[8 + n + a:8 + n + b]
        if v["dtype"] == "BF16":
            arr = (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(buf, {"F32": np.float32, "F64": np.float64, "I64": np.int64}[v["dtype"]])
        out[k] = arr.reshape(v["shape"])
    return out


@needs_goldens
@pytest.mark.parametrize("case,seed,lh,lw", [("t2i_2048", 42, 128, 128), ("t2i_3552x4736", 7, 222, 296)])
def test_t2i_noise(case, seed, lh, lw):
    target, _, _ = host.request_noise(seed, lh, lw)
    ref = _golden(f"{GOLDENS}/{case}/call0.safetensors")["hidden_states"][0]
    assert np.abs(target - ref).max() < 1e-5


@needs_goldens
def test_edit_front_end():
    from PIL import Image
    from tokenizers import Tokenizer
    front = f"{GOLDENS}/edit_4096/front"
    vae_img, vis_img = host.load_reference(Image.open(f"{MODEL}/examples/reference1.png"))
    pix, grid = host.vision_pixels(vis_img)
    te = _golden(f"{front}/te_inputs.safetensors")
    ref_pix = te["pixel_values"]
    assert tuple(te["image_grid_thw"][0][1:]) == grid
    assert np.abs(pix - ref_pix).max() <= 2**-7   # bf16 rounding of the reference
    tok = Tokenizer.from_file(f"{MODEL}/tokenizer/tokenizer.json")
    ids = host.token_ids(tok, Path(f"{MODEL}/examples/prompt.edit.txt").read_text(), [grid])
    assert ids.tolist() == te["input_ids"][0].tolist()
    assert np.abs(host.vae_pixels(vae_img) - _golden(f"{front}/vae_encode1.safetensors")["pixels"][0]).max() < 1e-5
    lh, lw = vae_img.height // 16, vae_img.width // 16
    target, eps, noise = host.request_noise(1234, 256, 256, [(lh, lw)])
    post = _golden(f"{front}/posterior_sample1.safetensors")["x"][0]
    enc = _golden(f"{front}/vae_encode1.safetensors")
    assert np.abs(enc["mean"][0] + enc["std"][0] * eps[0] - post).max() < 1e-4
    assert np.abs(noise[0] - _golden(f"{front}/ref_scale_noise6.safetensors")["noise"][0]).max() < 1e-5
    rows = _golden(f"{GOLDENS}/edit_4096/call0.safetensors")["hidden_states"][0][lh // 2 * (lw // 2):]
    assert np.abs(target - rows).max() < 1e-5
