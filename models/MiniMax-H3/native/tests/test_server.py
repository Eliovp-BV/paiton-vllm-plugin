"""HTTP API without a GPU: request checks, multipart and JSON parsing, and one request through a fake runner."""
import base64
import io
import json
from pathlib import Path
import sys
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paiton_h3v"))
import server  # noqa: E402


def png(color=(200, 10, 10), size=(64, 48)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("fields, images, videos, message", [
    ({}, [b"x"], [], "prompt"),
    ({"prompt": "a"}, [], [], "1 to 9"),
    ({"prompt": "a"}, [b"x"] * 10, [], "1 to 9"),
    ({"prompt": "a"}, [b"x"], [b"v", b"v"], "at most one video"),
    ({"prompt": "a", "width": 900}, [b"x"], [], "multiples of 32"),
    ({"prompt": "a", "width": 1920, "height": 1088}, [b"x"], [], "multiples of 32"),
    ({"prompt": "a", "num_frames": 125}, [b"x"], [], "17 n"),
    ({"prompt": "a", "num_frames": 379}, [b"x"], [], "17 n"),
    ({"prompt": "a", "preset": "turbo8"}, [b"x"], [], "preset turbo4"),
    ({"prompt": "a", "seed": -1}, [b"x"], [], "seed"),
])
def test_check_refuses(fields, images, videos, message):
    with pytest.raises(ValueError, match=message):
        server.check(fields, images, videos, "turbo4")


def test_check_defaults_and_seconds():
    assert server.check({"prompt": "a"}, [b"x"] * 9, [], "turbo4") == ("a", 864, 480, 124, 0)
    assert server.check({"prompt": "a", "seconds": "15", "width": "1344", "height": "768"}, [b"x"], [b"v"], "turbo4")[1:4] == (1344, 768, 362)


def test_parse_multipart_and_json():
    boundary = "b0undary"
    parts = [("prompt", None, b"two cats"), ("image", "a.png", b"AAA"), ("image", "b.png", b"BBB"), ("seed", None, b"7")]
    body = b"".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{n}"{f"; filename={chr(34)}{f}{chr(34)}" if f else ""}\r\n\r\n'.encode()
        + data + b"\r\n" for n, f, data in parts) + f"--{boundary}--\r\n".encode()
    fields, images, videos = server.parse(f"multipart/form-data; boundary={boundary}", body)
    assert fields == {"prompt": "two cats", "seed": "7"} and images == [b"AAA", b"BBB"] and videos == []
    body = json.dumps({"prompt": "p", "images": [base64.b64encode(b"AAA").decode()], "video": base64.b64encode(b"V").decode()})
    assert server.parse("application/json", body.encode()) == ({"prompt": "p"}, [b"AAA"], [b"V"])
    with pytest.raises(ValueError, match="base64"):
        server.parse("application/json", b'{"prompt": "p", "images": ["@@"]}')


def test_request_through_fake_runner(monkeypatch):
    seen = {}

    class FakeRunner:
        def __init__(self, command):
            seen["command"] = command
            self.proc = type("P", (), {"poll": lambda self: None})()

        def run(self, line):
            out = line.split("--out ")[1].strip()
            Path(out + ".mp4").write_bytes(b"MP4DATA")
            return {"request_s": "1.0"}

    def fake_line(home, preset, req, work):
        seen["req"] = req
        assert all(Image.open(r).size == (64, 48) for r in req["refs"])
        return "--ids x --out " + req["out"][:-4]

    monkeypatch.setattr(server, "Runner", FakeRunner)
    monkeypatch.setattr(server.h3v, "runner_cmd", lambda home, preset, extra: ["h3v_run", preset, *extra])
    monkeypatch.setattr(server.h3v, "request_line", fake_line)
    engine = server.Engine("/home", "turbo4", [])
    assert seen["command"] == ["h3v_run", "turbo4", "--serve"]
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), None)
    httpd.server_close()
    port = httpd.server_address[1]
    thread = threading.Thread(target=server.serve, args=(engine, "127.0.0.1", port), daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            health = json.load(urllib.request.urlopen(base + "/health"))
            break
        except OSError:
            threading.Event().wait(0.05)
    assert health["status"] == "ready" and health["max_references"] == 9
    body = json.dumps({"prompt": "a red card", "images": [base64.b64encode(png()).decode()] * 2, "seed": 3}).encode()
    reply = urllib.request.urlopen(urllib.request.Request(base + "/v1/videos/generations", body, {"Content-Type": "application/json"}))
    assert reply.read() == b"MP4DATA" and reply.headers["Content-Type"] == "video/mp4"
    assert json.loads(reply.headers["X-Paiton-Metrics"])["seed"] == "3"
    assert seen["req"]["num_frames"] == 124 and len(seen["req"]["refs"]) == 2
    bad = json.dumps({"prompt": "a", "images": [base64.b64encode(b"not an image").decode()]}).encode()
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(urllib.request.Request(base + "/v1/videos/generations", bad, {"Content-Type": "application/json"}))
    assert error.value.code == 400


def test_comfyui_node_payload_parses():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comfyui"))
    import numpy as np
    import paiton_h3v_client as node
    images = np.zeros((2, 48, 64, 3), np.float32)
    images[1, ..., 0] = 1.0
    pngs = node.to_png(images)
    assert len(pngs) == 2 and Image.open(io.BytesIO(pngs[1])).getpixel((0, 0)) == (255, 0, 0)
    body, content_type = node.multipart(dict(prompt="p", width=864, height=480, seconds=5.0, seed=2), pngs, b"VID")
    fields, got, videos = server.parse(content_type, body)
    assert got == pngs and videos == [b"VID"]
    assert server.check(fields, got, videos, "turbo4") == ("p", 864, 480, 124, 2)
