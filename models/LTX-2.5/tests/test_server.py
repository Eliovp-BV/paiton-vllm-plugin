"""Request parsing checks (no GPU). Run: python3 tests/test_server.py (or pytest)."""
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paiton_ltx25 import server  # noqa: E402


def raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__}{args} did not raise")


def test_fields():
    assert server._size("1280x704") == (1280, 704) and server._size("1920x1088") == (1920, 1088)
    for bad in ("1280", "1280x720", "1920x1152", "704x1280", "128x128", "ax b"):
        raises(server._size, bad)
    assert server._frames(121) == 121 and server._frames("9") == 9
    for bad in (120, 1, 129):
        raises(server._frames, bad)
    raises(server._seed, -1)
    raises(server._common, {"response_format": "url"})


def test_request_bodies():
    png = b"\x89PNG fake"
    body = json.dumps(dict(prompt="a cat", size="1920x1088", num_frames=97, seed=7, image=base64.b64encode(png).decode()))
    assert server.parse_request("application/json", body.encode()) == ("a cat", "1920x1088", 97, 7, png)
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    assert server.parse_request("application/json", json.dumps(dict(prompt="x", image=url)).encode())[4] == png
    assert server.parse_request("application/json", b'{"prompt": "x"}') == ("x", "1280x704", 121, 42, None)
    raises(server.parse_request, "application/json", b'{"prompt": "x", "image": "%%%"}')
    b = "ltx-x"
    multi = (f'--{b}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nwaves\r\n'
             f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="f.png"\r\nContent-Type: image/png\r\n\r\n').encode()
    multi += png + f"\r\n--{b}--\r\n".encode()
    assert server.parse_request(f"multipart/form-data; boundary={b}", multi) == ("waves", "1280x704", 121, 42, png)


def test_request_line_and_tokens():
    m = server.parse_request_line("REQUEST 0: 18.96 s | te 0.45 s1 5.89 up 0.11 s2 9.38 vae 2.74 audio 0.23 write 0.17 "
                                  "| VRAM 24.64 GiB, peak 25.98 GiB | prev mux 0.00 s")
    assert m == dict(runner_s=18.96, te_s=0.45, s1_s=5.89, up_s=0.11, s2_s=9.38, vae_s=2.74, audio_s=0.23, write_s=0.17,
                     vram_gib=24.64, vram_peak_gib=25.98)

    class Tok:
        def __init__(self, ids):
            self.ids = ids

        def encode(self, text):
            return self

    assert list(server.token_ids(Tok([5, 6]), " x ")) == [2, 5, 6]
    assert list(server.token_ids(Tok([2, 5]), "x")) == [2, 5]
    assert len(server.token_ids(Tok(list(range(3, 2000))), "x")) == 1024
    raises(server.token_ids, Tok([1]), "  ")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
