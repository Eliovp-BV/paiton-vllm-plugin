"""Host side of a MiniMax-H3-Image request: token ids, reference images and seed noise, numpy + PIL + tokenizers only.

Follows the diffusers modular pipeline (minimax_h3_image: before_encoder, encoders, before_denoise, decoders).
"""
import math

import numpy as np
from PIL import Image, ImageOps

CANVAS = 32                      # canvas sides are multiples of 32 (VAE 16 x patch 2)
MIN_AREA = 2048 * 2048           # smaller requests are generated on a larger canvas and resized back
MAX_TOKENS = 4096                # prompt and each reference label keep at most this many tokens
VISION_START, VISION_END, IMAGE_PAD = 151652, 151653, 151655
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def canvas_size(height, width):
    """(height, width) of the canvas a height x width image is generated on."""
    scale = max(1.0, math.sqrt(MIN_AREA / (width * height)))
    return (max(CANVAS, round(height * scale / CANVAS) * CANVAS), max(CANVAS, round(width * scale / CANVAS) * CANVAS))


def bound_short_edge(width, height, lo, hi, multiple=CANVAS):
    short = min(width, height)
    if lo <= short <= hi:
        return width, height
    scale = (lo if short < lo else hi) / short
    return max(multiple, round(width * scale / multiple) * multiple), max(multiple, round(height * scale / multiple) * multiple)


def _resize(image, size):
    return image if image.size == size else image.resize(size, Image.LANCZOS)


def smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=16777216):
    """Qwen2-VL image processor size rule (patch 16 x merge 2)."""
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = max(factor, math.floor(height / beta / factor) * factor), max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w


def vision_pixels(image):
    """Text-encoder copy of a reference -> (pixel_values f32 [GH*GW][1536], (GH, GW))."""
    h, w = smart_resize(image.height, image.width)
    if (w, h) != image.size:
        # shortcut: PIL bicubic, the reference processor uses torchvision's antialiased bicubic; only hit when a reference
        # keeps a short edge of exactly 1024 with a long edge that is not a multiple of 32
        image = image.resize((w, h), Image.BICUBIC)
    x = (np.asarray(image, np.float32) - 127.5) / 127.5            # rescale 1/255 + normalize mean 0.5 std 0.5, fused
    x = np.repeat(x.transpose(2, 0, 1)[None], 2, axis=0)          # [temporal 2][C][H][W]
    gh, gw = h // 16, w // 16
    x = x.reshape(1, 2, 3, gh // 2, 2, 16, gw // 2, 2, 16).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(x.reshape(gh * gw, 1536)), (gh, gw)


def vae_pixels(image):
    """VAE copy of a reference -> ImageNet-normalized f32 [3][H][W], cropped top-left to multiples of 16."""
    x = np.asarray(image, np.float32)[: image.height // 16 * 16, : image.width // 16 * 16] / 255.0
    return np.ascontiguousarray(((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))


def load_reference(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) < CANVAS:
        raise ValueError(f"A reference image needs at least {CANVAS} pixels on each side, got {image.size}")
    return _resize(image, bound_short_edge(*image.size, 1024, 2048)), _resize(image, bound_short_edge(*image.size, 1024, 1024))


def token_ids(tokenizer, prompt, vision_grids=()):
    """Edit presentation: per reference 'Picture i: ' + vision block, then the prompt; no template, no special tokens."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    ids = []
    for i, (gh, gw) in enumerate(vision_grids):
        ids += tokenizer.encode(f"Picture {i + 1}: ", add_special_tokens=False).ids[:MAX_TOKENS]
        ids += [VISION_START] + [IMAGE_PAD] * (gh * gw // 4) + [VISION_END]
    return np.array(ids + tokenizer.encode(prompt, add_special_tokens=False).ids[:MAX_TOKENS], np.int32)


class TorchCpuGenerator:
    """torch.Generator('cpu').manual_seed(seed) + torch.randn(float32), the mt19937 Box-Muller path (numel >= 16)."""

    def __init__(self, seed):
        self.bits = np.random.MT19937()
        self.bits._legacy_seeding(seed & 0xFFFFFFFF)

    def _uniform(self, n):
        return (self.bits.random_raw(n) & 0xFFFFFF).astype(np.float32) * np.float32(2.0**-24)

    @staticmethod
    def _box_muller(u):   # u: [k][16]; first 8 lanes become radius*cos, last 8 radius*sin
        u1, u2 = 1.0 - u[:, :8].astype(np.float64), u[:, 8:].astype(np.float64)
        r, th = np.sqrt(-2.0 * np.log(u1)), 2.0 * np.pi * u2
        return np.concatenate([r * np.cos(th), r * np.sin(th)], axis=1).astype(np.float32)

    def randn(self, n):
        if n < 16:
            raise ValueError("small randn uses another torch path")   # never hit: latents have >= 24 * 4 values
        data = self._uniform(n)
        full = n // 16 * 16
        data[:full] = self._box_muller(data[:full].reshape(-1, 16)).reshape(-1)
        if n % 16:   # torch redraws the last 16 values and transforms them again
            data[n - 16:] = self._uniform(16)
            data[n - 16:] = self._box_muller(data[n - 16:].reshape(1, 16)).reshape(-1)
        return data


def patchify(z, lh, lw):
    """[24][lh][lw] -> rows [(lh/2)*(lw/2)][96], channel-major then (dy, dx)."""
    return np.ascontiguousarray(z.reshape(24, lh // 2, 2, lw // 2, 2).transpose(1, 3, 0, 2, 4).reshape(-1, 96))


def request_noise(seed, lh, lw, ref_grids=()):
    """Draws in the reference order: one posterior eps per reference, the target noise, one noise per reference."""
    gen = TorchCpuGenerator(seed)
    eps = [gen.randn(24 * h * w).reshape(24, h, w) for h, w in ref_grids]
    target = patchify(gen.randn(24 * lh * lw), lh, lw)
    noise = [gen.randn(24 * h * w).reshape(24, h, w) for h, w in ref_grids]
    return target, eps, noise


def unscale(png_image, height, width):
    """Resize a larger-canvas result back to the requested size (diffusers VaeImageProcessor.resize, Lanczos)."""
    return _resize(png_image, (width, height))
