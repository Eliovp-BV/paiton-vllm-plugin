"""Start-time item A6, read-ahead half (opt-in: PAITON_START_READAHEAD=1).

While the API server spends its first ~45 s importing Python, the disk is mostly idle. This helper, spawned detached by
the serving CLI just before it hands over to vLLM, asks the kernel to read ahead the bytes the engine will stream first:
the target tensors the W3 build does not replace (the adapter's pre_read_skip predicate, when the installed adapter has
it; otherwise the whole checkpoint is left alone) and the drafter checkpoint. POSIX_FADV_WILLNEED is asynchronous and
advisory: no bytes change, nothing is held in memory by this process, and it is skipped when the host could not keep the
bytes cached anyway (MemAvailable below the advised size plus a 4 GiB margin for the imports).
"""
import glob
import json
import os
import struct
import sys


def mem_available_bytes():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return 0


def predicate(language_model_only):
    try:
        import paiton_w3_decode_adapter as adapter   # auto-imported by the image's .pth hook when W3 is on
    except Exception:  # noqa: BLE001  (W3 off or adapter refused: advise nothing of the target)
        return None
    factory = getattr(adapter, 'pre_read_skip', None)
    try:
        return factory(language_model_only) if factory else None
    except Exception:  # noqa: BLE001
        return None


def ranges(target, skip):
    out = []
    for path in sorted(glob.glob(os.path.join(target, '*.safetensors'))):
        with open(path, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == '__metadata__' or skip(name):
                continue
            a, b = meta['data_offsets']
            out.append((path, 8 + n + a, b - a))
    return out


def advise(items):
    total = 0
    by_file = {}
    for path, off, length in items:
        by_file.setdefault(path, []).append((off, length))
    for path, segs in by_file.items():
        fd = os.open(path, os.O_RDONLY)
        try:
            for off, length in sorted(segs):
                os.posix_fadvise(fd, off, length, os.POSIX_FADV_WILLNEED)
                total += length
        finally:
            os.close(fd)
    return total


def main(argv):
    target, draft = argv[1], argv[2]
    language_model_only = '--language-model-only' in argv
    os.nice(19)
    skip = predicate(language_model_only)
    items = ranges(target, skip) if skip is not None else []
    for path in sorted(glob.glob(os.path.join(draft, '*.safetensors'))):
        items.append((path, 0, os.stat(path).st_size))
    want = sum(length for _, _, length in items)
    avail = mem_available_bytes()
    if want + 4 * 2 ** 30 > avail:
        print(f'[paiton.readahead] skipped: {want / 2 ** 30:.2f} GiB wanted, {avail / 2 ** 30:.2f} GiB available', flush=True)
        return 0
    total = advise(items)
    print(f'[paiton.readahead] advised {total / 2 ** 30:.2f} GiB ({"target kept ranges + " if skip else ""}drafter)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
