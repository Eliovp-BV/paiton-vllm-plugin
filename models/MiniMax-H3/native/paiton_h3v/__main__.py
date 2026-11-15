"""Container entry point: check the weights, lay out H3V_HOME, then hand over to h3v.py.

    python3 -m paiton_h3v prepare [--preset turbo8]     download and check the weights, then exit
    python3 -m paiton_h3v generate --prompt .. --ref a.png --out clip.mp4 [--preset turbo4]
    python3 -m paiton_h3v serve [--preset turbo4]       one JSON request per stdin line (see h3v.py)
    python3 -m paiton_h3v http [--preset turbo4] [--port 30000]   HTTP API (see server.py)
"""
import os
from pathlib import Path
import sys

from . import weights

APP = Path(__file__).resolve().parent.parent
HOME_ITEMS = ("weights", "te-w8", "dec8.bin", "enc.bin", "audio.bin", "model")


def make_home(bundle):
    """The runner expects one folder: libs/ from the image, everything else from the downloaded bundle."""
    home = Path(os.environ.get("TMPDIR", "/tmp")) / "h3v-home"
    home.mkdir(parents=True, exist_ok=True)
    for name, target in [("libs", APP / "lib")] + [(n, bundle / n) for n in HOME_ITEMS]:
        link = home / name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target)
    return home


def main():
    args = sys.argv[1:] or ["serve"]
    preset = args[args.index("--preset") + 1] if "--preset" in args[:-1] else "turbo4"
    try:
        bundle = weights.prepare(Path(os.environ.get("PAITON_CACHE_DIR", "/cache")), preset)
    except (ValueError, OSError) as error:
        sys.exit(f"weights: {error}")
    if args[0] == "prepare":
        return
    os.environ["H3V_HOME"] = str(make_home(bundle))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args[0] == "http":
        import server   # noqa: E402
        return server.main(args[1:])
    import h3v   # noqa: E402  (h3v.py imports front.py from the same folder)
    sys.argv = ["h3v.py", *args]
    h3v.main()


if __name__ == "__main__":
    main()
