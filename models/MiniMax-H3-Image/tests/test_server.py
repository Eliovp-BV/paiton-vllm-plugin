"""Request parsing checks (no GPU)."""
import io
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paiton_h3i import server  # noqa: E402


def test_multipart_matches_example_client():
    png = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 20, 30)).save(png, "PNG")
    b = "minimax-h3-image-x"
    body = (f'--{b}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nMake it blue\r\n'
            f'--{b}\r\nContent-Disposition: form-data; name="seed"\r\n\r\n1234\r\n'
            f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="r.png"\r\nContent-Type: image/png\r\n\r\n').encode()
    body += png.getvalue() + f"\r\n--{b}--\r\n".encode()
    fields, images = server._multipart(f"multipart/form-data; boundary={b}", body)
    assert fields == {"prompt": "Make it blue", "seed": "1234"}
    assert len(images) == 1 and images[0].size == (64, 48)


def test_fields():
    assert server._size("4736x3552") == (4736, 3552)
    for bad in ("4096", "0x10", "ax b"):
        with pytest.raises(ValueError):
            server._size(bad)
    with pytest.raises(ValueError):
        server._common({"response_format": "url"})
    with pytest.raises(ValueError):
        server._seed(-1)
