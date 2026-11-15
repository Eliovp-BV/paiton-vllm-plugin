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
    assert server._size("1280x704") == (1280, 704) and server._size("1088x1920") == (1088, 1920)
    for bad in ("1280", "1280x720", "1920x1152", "128x128", "ax b"):
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
    m = server.parse_request_line("REQUEST 0: 23.34 s | te 0.50 s1 6.50 up 0.10 s2 11.42 vae 2.72 audio 0.22 write 1.77 | VRAM 27.51 GiB")
    assert m == dict(runner_s=23.34, te_s=0.5, s1_s=6.5, up_s=0.1, s2_s=11.42, vae_s=2.72, audio_s=0.22, write_s=1.77, vram_gib=27.51)

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
