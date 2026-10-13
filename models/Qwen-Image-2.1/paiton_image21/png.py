"""Lossless PNG encoder for the image engine's RGBA output (replaces the single-threaded PIL encoder on the request path): per-row Sub/Up filter computed with numpy, the zlib stream
compressed in parallel chunks (raw deflate per chunk with a sync flush, pigz layout: one valid zlib stream), written
as a standard 8-bit RGBA PNG. Decoded pixels are identical to the input; only the compressed size differs."""
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_POOL = None

def _chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(data, zlib.crc32(tag)) & 0xffffffff)

def _deflate(args):
    data, level, last = args
    c = zlib.compressobj(level, zlib.DEFLATED, -15, 9)
    return c.compress(data) + c.flush(zlib.Z_FINISH if last else zlib.Z_SYNC_FLUSH)

def encode_png(rgba, level=1, threads=8, filter_type=1, rows_per_chunk=None):
    """rgba: uint8 numpy array [H][W][4] (C-contiguous). filter_type 1 = Sub, 2 = Up, 0 = None."""
    global _POOL
    a = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w, c = a.shape
    assert c == 4
    raw = np.empty((h, 1 + w * 4), dtype=np.uint8)
    raw[:, 0] = filter_type
    flat = a.reshape(h, w * 4)
    if filter_type == 1:
        raw[:, 1:5] = flat[:, :4]
        np.subtract(flat[:, 4:], flat[:, :-4], out=raw[:, 5:])     # uint8 wraps modulo 256 as the PNG filter defines
    elif filter_type == 2:
        raw[0, 1:] = flat[0]
        np.subtract(flat[1:], flat[:-1], out=raw[1:, 1:])
    else:
        raw[:, 1:] = flat
    buf = raw.reshape(-1)
    rows_per_chunk = rows_per_chunk or max(16, -(-h // (threads * 2)))
    step = rows_per_chunk * raw.shape[1]
    parts = [(buf[i:i + step].tobytes(), level, i + step >= buf.size) for i in range(0, buf.size, step)]
    if threads > 1:
        if _POOL is None or _POOL._max_workers != threads:
            _POOL = ThreadPoolExecutor(threads)
        bodies = list(_POOL.map(_deflate, parts))
    else:
        bodies = [_deflate(p) for p in parts]
    adler = zlib.adler32(buf)
    stream = b"\x78\x01" + b"".join(bodies) + struct.pack(">I", adler & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", stream) + _chunk(b"IEND", b"")
