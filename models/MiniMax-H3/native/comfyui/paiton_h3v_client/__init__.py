"""ComfyUI client node for the Paiton MiniMax H3 Ref2VA v1.2 HTTP server (`launch.sh http`). It sends the prompt and up to
nine reference images to the server and returns the MP4 as a VIDEO. No model code runs inside ComfyUI."""
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image

MAX_REFS = 9


def multipart(fields, images, video=None):
    """(body, content type) of a multipart/form-data request: text fields, PNG `image` parts, optional `video` part."""
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for i, data in enumerate(images):
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="ref{i}.png"\r\n'
                  f'Content-Type: image/png\r\n\r\n'.encode() + data + b"\r\n")
    if video:
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="ref.mp4"\r\n'
                  f'Content-Type: video/mp4\r\n\r\n'.encode() + video + b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def to_png(image):
    """ComfyUI IMAGE (float [B][H][W][C] in 0..1) -> one PNG per batch entry."""
    pngs = []
    for frame in np.asarray(image.cpu() if hasattr(image, "cpu") else image):
        buffer = io.BytesIO()
        Image.fromarray(np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8)[..., :3]).save(buffer, "PNG")
        pngs.append(buffer.getvalue())
    return pngs


def generate(server, fields, pngs, video=None, timeout=1800):
    body, content_type = multipart(fields, pngs, video)
    request = urllib.request.Request(server.rstrip("/") + "/v1/videos/generations", body, {"Content-Type": content_type})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return reply.read(), json.loads(reply.headers.get("X-Paiton-Metrics", "{}"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Paiton server: {error.read().decode(errors='replace')}") from None


class PaitonH3Ref2VA:
    CATEGORY = "Paiton/MiniMax H3"
    FUNCTION = "run"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "metrics")
    DESCRIPTION = "MiniMax H3 Ref2VA on the Paiton v1.2 server: prompt + 1 to 9 reference images -> video with audio."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "reference images; a batch counts as several references"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 864, "min": 256, "max": 1344, "step": 32}),
                "height": ("INT", {"default": 480, "min": 256, "max": 1344, "step": 32}),
                "seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
                "server": ("STRING", {"default": "http://127.0.0.1:30000"}),
            },
            "optional": {f"image_{i}": ("IMAGE",) for i in range(2, MAX_REFS + 1)},
        }

    def run(self, images, prompt, width, height, seconds, seed, server, **more):
        pngs = to_png(images)
        for i in range(2, MAX_REFS + 1):
            if more.get(f"image_{i}") is not None:
                pngs += to_png(more[f"image_{i}"])
        if len(pngs) > MAX_REFS:
            raise ValueError(f"{len(pngs)} reference images; the server takes at most {MAX_REFS}")
        mp4, metrics = generate(server, dict(prompt=prompt, width=width, height=height, seconds=seconds, seed=seed), pngs)
        import folder_paths   # ComfyUI modules: imported here so the helpers above work without ComfyUI
        from comfy_api.input_impl import VideoFromFile
        path = os.path.join(folder_paths.get_temp_directory(), f"paiton_h3v_{int(time.time() * 1000)}.mp4")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as out:
            out.write(mp4)
        return VideoFromFile(path), json.dumps(metrics)


NODE_CLASS_MAPPINGS = {"PaitonH3Ref2VA": PaitonH3Ref2VA}
NODE_DISPLAY_NAME_MAPPINGS = {"PaitonH3Ref2VA": "MiniMax H3 Ref2VA (Paiton server)"}
