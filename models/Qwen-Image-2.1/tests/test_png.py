"""The request-path PNG encoder is lossless: decoded pixels equal the input for every filter, level and chunking."""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paiton_image21.png import encode_png


def _roundtrip(a, **kw):
    data = encode_png(a, **kw)
    img = Image.open(io.BytesIO(data))
    img.load()
    assert img.mode == "RGBA" and img.size == (a.shape[1], a.shape[0])
    return np.asarray(img)


def test_lossless_random_and_smooth():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (67, 129, 4), dtype=np.uint8)
    y, x = np.mgrid[0:96, 0:160]
    smooth = np.stack([x % 256, y % 256, (x + y) % 256, np.full_like(x, 200)], -1).astype(np.uint8)
    for a in (noise, smooth):
        for filter_type in (0, 1, 2):
            for level in (1, 6):
                for threads, rows in ((1, None), (3, 7), (8, 1)):
                    assert np.array_equal(_roundtrip(a, level=level, threads=threads, filter_type=filter_type, rows_per_chunk=rows), a)


def test_single_row_and_alpha():
    a = np.zeros((1, 5, 4), dtype=np.uint8)
    a[0, :, 3] = [0, 1, 127, 254, 255]
    assert np.array_equal(_roundtrip(a, threads=4), a)
