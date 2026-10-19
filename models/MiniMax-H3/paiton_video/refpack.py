"""MiniMax H3 Ref2VA packed-sequence planning.

Mirrors ComfyUI's ``PackedLayout`` for the ``[text | refs | audio | video]``
document: reference rows are re-injected every step, positions are float64
``(t, h, w)`` and the shared RoPE clock advances in presentation order.
"""
from dataclasses import dataclass
import math
from typing import ClassVar

import torch


CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768*1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0/3.0
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0
DEFAULT_MAX_TOKENS = 49152
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3


@dataclass(frozen=True)
class RefImage:
    kind: ClassVar[str] = "image"
    index: int
    image: object


@dataclass(frozen=True)
class RefVideo:
    kind: ClassVar[str] = "video"
    index: int
    frames: object
    audio: object = None


@dataclass(frozen=True)
class RefAudio:
    kind: ClassVar[str] = "audio"
    index: int
    audio: object


@dataclass(frozen=True)
class RefRequest:
    images: tuple = ()
    videos: tuple = ()
    audios: tuple = ()

    @property
    def order(self):
        """Tokenizer presentation order: images, soundtrack before video, audio."""
        result = [item for item in self.images]
        for video in self.videos:
            if video.audio is not None:
                result.append(RefAudio(video.index, video.audio))
            result.append(video)
        result.extend(self.audios)
        return tuple(result)

    def blocks(self):
        """DiT payload order: a video and its soundtrack are one block."""
        result = [{"kind": "image", "index": item.index, "image": item.image}
                  for item in self.images]
        for item in self.videos:
            block = {"kind": "video_audio" if item.audio is not None else "video",
                     "index": item.index, "frames": item.frames, "audio": item.audio}
            result.append(block)
        result.extend({"kind": "audio", "index": item.index, "audio": item.audio}
                      for item in self.audios)
        return result


@dataclass(frozen=True)
class PackedRefLayout:
    seq_len: int
    segments: tuple
    position_ids: torch.Tensor
    rope: torch.Tensor
    token_tags: torch.Tensor
    timestep_indices: torch.Tensor
    timestep_values: torch.Tensor
    adaln_indices: torch.Tensor
    text_indices: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    img_update: torch.Tensor
    audio_update: torch.Tensor
    signature: tuple

    @property
    def img_pos(self):
        return self.video_indices

    @property
    def audio_pos(self):
        return self.audio_indices


def _ordered_values(values, prefix, limit, label):
    if values is None:
        return []
    if isinstance(values, dict):
        items = []
        for key, value in values.items():
            if value is None:
                continue
            if not key.startswith(prefix) or not key[len(prefix):].isdigit():
                raise ValueError(f"Unexpected H3 reference input: {key}")
            items.append((int(key[len(prefix):]), value))
        items.sort()
    else:
        items = [(index, value) for index, value in enumerate(values) if value is not None]
    if len(items) > limit:
        raise ValueError(f"H3 Ref2VA accepts at most {limit} {label}")
    return items


def _first_frame(image, label):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] < 1:
        raise ValueError(f"{label} must be a batched IMAGE [B, H, W, C]")
    # ComfyUI's reference loop silently collapses a batched image to its first frame.
    return image[:1]


def parse_refs(ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
    images = tuple(RefImage(index, _first_frame(image, f"ref_image_{index}"))
                   for index, image in _ordered_values(ref_images, "ref_image_", MAX_REF_IMAGES, "images"))
    videos = []
    for index, frames in _ordered_values(ref_videos, "ref_video_", MAX_REF_VIDEOS, "videos"):
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4 or frames.shape[0] < 1:
            raise ValueError(f"ref_video_{index} must be a batched IMAGE [F, H, W, C]")
        videos.append(RefVideo(index, frames, None))
    soundtracks = dict(_ordered_values(ref_video_audios, "ref_video_audio_", MAX_REF_VIDEOS, "video soundtracks"))
    videos = [RefVideo(item.index, item.frames, soundtracks.get(item.index)) for item in videos]
    audios = tuple(RefAudio(index, audio)
                   for index, audio in _ordered_values(ref_audios, "ref_audio_", MAX_REF_AUDIOS, "audios"))
    return RefRequest(images, tuple(videos), audios)


def plan_ref_geometry(target_width, target_height, mode, width, height):
    """Reference-image canvas: downscale-only, rounded to a multiple of 32."""
    if mode == "match":
        scale = min(1.0, math.sqrt((target_width*target_height)/(width*height)))
    elif mode == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE/min(width, height))
    else:
        raise ValueError(f"Unsupported ref_image_size: {mode}")
    return (max(CANVAS_MULTIPLE, round(width*scale/CANVAS_MULTIPLE)*CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(height*scale/CANVAS_MULTIPLE)*CANVAS_MULTIPLE))


def plan_video_geometry(width, height):
    """Reference-video canvas: 768 short edge / 768*1344 area cap, own size if smaller."""
    ratio = width/height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE*ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE/ratio
    if nom_w*nom_h > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS/(nom_w*nom_h))
        nom_w, nom_h = nom_w*scale, nom_h*scale
    canvas = (max(CANVAS_MULTIPLE, round(nom_w/CANVAS_MULTIPLE)*CANVAS_MULTIPLE),
              max(CANVAS_MULTIPLE, round(nom_h/CANVAS_MULTIPLE)*CANVAS_MULTIPLE))
    if width*height < canvas[0]*canvas[1]:
        canvas = (max(CANVAS_MULTIPLE, round(width/CANVAS_MULTIPLE)*CANVAS_MULTIPLE),
                  max(CANVAS_MULTIPLE, round(height/CANVAS_MULTIPLE)*CANVAS_MULTIPLE))
    return canvas


def trim_video_frames(frame_count, target_frames):
    count = min(int(frame_count), int(target_frames))
    if count < 5:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
    while count % 17 != 5:
        count -= 1
    return count


def ref_timestep(timestep):
    return max(float(timestep), VISUAL_COND_TIMESTEP)


def ref_audio_timestep(timestep):
    return max(float(timestep), AUDIO_COND_TIMESTEP)


def _block_dims(block):
    kind = block["kind"]
    if kind == "image":
        h = int(block["latent_h"]) if "latent_h" in block else int(block["latent"].shape[-2])
        w = int(block["latent_w"]) if "latent_w" in block else int(block["latent"].shape[-1])
        return 0, h, w, 0
    if kind in ("video", "video_audio"):
        t = int(block["latent_t"]) if "latent_t" in block else int(block["latent"].shape[2])
        h = int(block["latent_h"]) if "latent_h" in block else int(block["latent"].shape[-2])
        w = int(block["latent_w"]) if "latent_w" in block else int(block["latent"].shape[-1])
        rt = int(block.get("ref_audio_t", 0))
        return t, h, w, rt
    if kind == "audio":
        rt = int(block["ref_audio_t"]) if "ref_audio_t" in block else int(block["audio_latent"].shape[-1])
        return 0, 0, 0, rt
    raise ValueError(f"Unsupported H3 reference kind: {kind}")


def _ref_t_span(block, dims):
    kind = block["kind"]
    if kind == "image":
        return 1.0
    if kind == "audio":
        return float(dims[3])
    if kind in ("video", "video_audio"):
        return max(float(dims[3]), sum(_video_t_spans(dims[0])))
    return 0.0


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    ratio = dim/sqrt_area
    n = dim//patch
    return (torch.arange(n, dtype=torch.float64)*(ratio/n) + (1.0-ratio)/2.0)*32.0


def _frame_grid(h, w):
    area = math.sqrt(h*w)
    hh, ww = torch.meshgrid(_axis_from_sqrt_area(h, 2, area),
                            _axis_from_sqrt_area(w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis_from_sqrt_area(w, 2, area)


def _video_t_spans(n):
    return [FRAME_RESCALE*FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n, origin):
    spans = torch.tensor(_video_t_spans(n), dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _audio_grid(cursor, t, w_low, w_high):
    grid = torch.zeros(t*2, 3, dtype=torch.float64)
    grid[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)
    grid[:t, 2] = w_low
    grid[t:, 2] = w_high
    return grid


def _video_grid(vt, frame, cursor):
    grid = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)
    grid[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    grid[:, :, 1:] = frame[None]
    return grid.reshape(-1, 3)


def _default_inv_freq():
    return 1.0/(10000.0**(torch.arange(16, dtype=torch.float32)/16.0))


def build_packed_layout(text_len, latent_t, latent_h, latent_w, audio_t, refs=None,
                        video_timestep=0.0, audio_timestep=None,
                        text_token_tags=None, max_tokens=DEFAULT_MAX_TOKENS, inv_freq=None,
                        adaln_values=None):
    """Build ``[text | refs | target audio | target video]`` for one step."""
    text_len, latent_t, latent_h, latent_w, audio_t = (
        int(text_len), int(latent_t), int(latent_h), int(latent_w), int(audio_t))
    if min(text_len, latent_t, latent_h, latent_w, audio_t) < 0:
        raise ValueError("Packed layout dimensions must be non-negative")
    if text_len < 1 or latent_t < 1 or latent_h < 2 or latent_w < 2 or audio_t < 1:
        raise ValueError("Packed layout dimensions are too small")
    if latent_h % 2 or latent_w % 2:
        raise ValueError("Packed video latents must be even in H and W")
    if audio_timestep is None:
        audio_timestep = video_timestep
    blocks = list(refs or ())
    dims = [_block_dims(block) for block in blocks]

    frame, w_grid = _frame_grid(latent_h, latent_w)
    frame_rows = frame.shape[0]
    target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
    target_cursor = float(text_len) + sum(_ref_t_span(block, dim) for block, dim in zip(blocks, dims))
    seq_len = text_len + sum(2*int(dim[3]) if block["kind"] in ("audio", "video", "video_audio") else 0
                             for block, dim in zip(blocks, dims))
    seq_len += sum(int(dim[0])*((dim[1]//2)*(dim[2]//2)) if block["kind"] in ("video", "video_audio")
                   else (dim[1]//2)*(dim[2]//2) if block["kind"] == "image" else 0
                   for block, dim in zip(blocks, dims))
    seq_len += audio_t*2 + latent_t*frame_rows
    if seq_len > max_tokens:
        raise ValueError(f"H3 packed sequence {seq_len} exceeds the {max_tokens}-token cap")

    segments = [("text", text_len)]
    text_pos = torch.zeros(text_len, 3, dtype=torch.float64)
    text_pos[:, 0] = torch.arange(text_len, dtype=torch.float64)
    positions = [text_pos]
    img_pos, img_update, audio_pos, audio_update = [], [], [], []
    row = text_len
    cursor = float(text_len)
    for block, (vt, h, w, rt) in zip(blocks, dims):
        kind = block["kind"]
        if kind == "image":
            ref_frame, _ = _frame_grid(h, w)
            n = ref_frame.shape[0]
            grid = torch.empty(n, 3, dtype=torch.float64)
            grid[:, 0] = cursor
            grid[:, 1:] = ref_frame
            segments.append(("ref_img", n))
            positions.append(grid)
            img_pos.append(torch.arange(row, row+n))
            img_update.append(torch.zeros(n, dtype=torch.bool))
            row += n
            cursor += 1.0
        elif kind == "audio":
            if rt > 0:
                segments.append(("ref_audio", rt*2))
                positions.append(_audio_grid(cursor, rt, *target_audio_w))
                audio_pos.append(torch.arange(row, row+rt*2))
                audio_update.append(torch.zeros(rt*2, dtype=torch.bool))
                row += rt*2
            cursor += float(rt)
        else:
            ref_frame, ref_w_grid = _frame_grid(h, w)
            if rt > 0:
                segments.append(("ref_audio", rt*2))
                positions.append(_audio_grid(cursor, rt, float(ref_w_grid[0]), float(ref_w_grid[-1])))
                audio_pos.append(torch.arange(row, row+rt*2))
                audio_update.append(torch.zeros(rt*2, dtype=torch.bool))
                row += rt*2
            n = vt*ref_frame.shape[0]
            segments.append(("ref_img", n))
            positions.append(_video_grid(vt, ref_frame, cursor))
            img_pos.append(torch.arange(row, row+n))
            img_update.append(torch.zeros(n, dtype=torch.bool))
            row += n
            cursor += max(float(rt), sum(_video_t_spans(vt)))

    if cursor != target_cursor:
        raise AssertionError(f"H3 reference clock mismatch: {cursor} != {target_cursor}")
    segments.append(("audio", audio_t*2))
    positions.append(_audio_grid(target_cursor, audio_t, *target_audio_w))
    audio_pos.append(torch.arange(row, row+audio_t*2))
    audio_update.append(torch.ones(audio_t*2, dtype=torch.bool))
    row += audio_t*2

    n_video = latent_t*frame_rows
    segments.append(("video", n_video))
    positions.append(_video_grid(latent_t, frame, target_cursor))
    img_pos.append(torch.arange(row, row+n_video))
    img_update.append(torch.ones(n_video, dtype=torch.bool))
    row += n_video
    if row != seq_len:
        raise AssertionError(f"H3 packed length mismatch: {row} != {seq_len}")

    position_ids = torch.cat(positions)
    if text_token_tags is None:
        tags = torch.ones(text_len, dtype=torch.long)
    else:
        tags = torch.as_tensor(text_token_tags, dtype=torch.long).reshape(-1)
        if tags.numel() != text_len or bool(((tags < 0) | (tags > 2)).any()):
            raise ValueError("text_token_tags must be one 0/1/2 tag per text row")
    absolute = []
    offset = 0
    for kind, count in segments:
        absolute.append((offset, offset+count, kind))
        offset += count
    segments = absolute
    token_tags = torch.empty(seq_len, dtype=torch.long)
    token_tags[:text_len] = tags
    seg_t = {"text": float(video_timestep), "video": float(video_timestep),
             "audio": float(audio_timestep), "ref_img": ref_timestep(video_timestep),
             "ref_audio": ref_audio_timestep(audio_timestep)}
    if adaln_values is None:
        values = sorted(set(seg_t.values()))
        t_row = {value: index for index, value in enumerate(values)}
        timestep_values = torch.tensor(values, dtype=torch.float32)
        signature_values = tuple(values)
    else:
        values = torch.as_tensor(adaln_values, dtype=torch.float32).reshape(-1)
        if not values.numel() or not bool(torch.isfinite(values).all()):
            raise ValueError("H3 AdaLN values must be a non-empty finite array")
        if bool((values[1:] < values[:-1]).any()):
            raise ValueError("H3 AdaLN values must be sorted ascending")
        def global_row(value):
            target = torch.tensor(float(value), dtype=torch.float32)
            index = int(torch.searchsorted(values, target).item())
            if index >= values.numel() or not bool(torch.isclose(values[index], target, atol=1e-6, rtol=0)):
                raise ValueError(f"H3 AdaLN table is missing timestep {float(value)}")
            return index
        t_row = {value: global_row(value) for value in set(seg_t.values())}
        timestep_values = values
        signature_values = tuple(float(value) for value in values.tolist())
    timestep_indices = torch.empty(seq_len, dtype=torch.long)
    tags_by_kind = {"text": 1, "video": 0, "audio": 2, "ref_img": 0, "ref_audio": 2}
    for start, stop, kind in segments:
        token_tags[start:stop] = tags_by_kind[kind] if kind != "text" else tags
        timestep_indices[start:stop] = t_row[seg_t[kind]]
    adaln_indices = timestep_indices*3 + token_tags

    inv = _default_inv_freq() if inv_freq is None else torch.as_tensor(inv_freq, dtype=torch.float32).reshape(-1)
    if inv.numel() != 16:
        raise ValueError("H3 RoPE requires 16 inverse frequencies")
    per_axis = position_ids.to(torch.float32).unsqueeze(-1)*inv.view(1, 1, -1)
    t_f, h_f, w_f = per_axis.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)
    rope = torch.cat((half, half), dim=-1)

    ref_signature = tuple((block["kind"],)+tuple(int(value) for value in dim)
                          for block, dim in zip(blocks, dims))
    signature = (text_len, latent_t, latent_h, latent_w, audio_t, ref_signature,
                 tuple(tags.tolist()), signature_values)
    return PackedRefLayout(
        seq_len=seq_len, segments=tuple(segments), position_ids=position_ids, rope=rope,
        token_tags=token_tags, timestep_indices=timestep_indices,
        timestep_values=timestep_values, adaln_indices=adaln_indices,
        text_indices=torch.arange(text_len),
        video_indices=torch.cat(img_pos), audio_indices=torch.cat(audio_pos),
        img_update=torch.cat(img_update), audio_update=torch.cat(audio_update),
        signature=signature)
