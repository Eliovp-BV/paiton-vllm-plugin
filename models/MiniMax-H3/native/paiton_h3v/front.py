"""Host front end of the Ref2VA runner: request -> runner input files + one request line. numpy, PIL, tokenizers; no torch.

    python3 front.py --model <MiniMax-H3-diffusers dir> --request request.json --out <dir> [--preset turbo4]
                     [--seed N] [--noise-from <golden dir>]

request.json: {"prompt": str, "width": int, "height": int, "num_frames": int, "refs": [image or video paths],
"ref_mode": "match"}
(the R0 golden request.json "case" block has this shape). Reproduces the diffusers Ref2VA pipeline host steps:
  - reference sizing: v1.1.0 `match` rule (downscale-only to the target area, multiple of 32, LANCZOS), as r0_goldens.py
  - Qwen3-VL image processor (smart_resize to multiples of 32, (x / 255 - 0.5) / 0.5, 16 px patches, temporal patch 2,
    2 x 2 merge order) -> pixels f32 [gh * gw][1536] per image reference
  - presentation tokens ("<Picture i>: " + vision block per reference, then the prompt) and their tags
  - packed layout (build_ref2va_packed_sequence): positions f64 [S][3], tags, video / audio row indices
  - sigma schedules (exponential shift, video 6, audio 3 for the turbo presets)
  - VAE pixels uint8 [F][H][W][3] per visual reference (the runner normalizes)
  - video references (.mp4 / .mov / .mkv / .webm): ffmpeg decode to rgb24, whole-frame resample to 24 fps, truncate to
    the generated frame count, LANCZOS onto the canvas of their own aspect ratio; the text encoder reads them at 2 fps,
    pairs of frames per vision block with a "<t seconds>" label; their soundtrack (ffmpeg, f32 stereo) is truncated to
    the generated duration and becomes the reference audio rows, packed right before the reference's video rows.
The runner's K4 text encoder knows one vision placeholder id, so video blocks are written with <|image_pad|>: the
placeholder embedding is replaced by the vision features either way, and Qwen3-VL places each video block (t = 1) like
an image block.
shortcut: no audio-only references, and soundtracks must already be at 32 kHz (no torchaudio resampler here).
"""
import subprocess
import argparse
import json
import os

import numpy as np
from PIL import Image
from tokenizers import Tokenizer

VISION_START, VISION_END, IMAGE_PAD = 151652, 151653, 151655
PIXEL_MEAN, PIXEL_STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
PRESETS = {'turbo4': 5, 'turbo8': 9}   # sigma grid points (terminal 0 included): 4 / 8 evaluations
SHIFT_VIDEO, SHIFT_AUDIO = 6.0, 3.0


VIDEO_EXT = ('.mp4', '.mov', '.mkv', '.webm')


def resolve_canvas(aw, ah, multiple=32, short_edge=768, max_pixels=768 * 1344):
    ratio = aw / ah
    assert 0.25 <= ratio <= 4, 'aspect ratio out of range'
    width, height = (short_edge * ratio, float(short_edge)) if ratio >= 1 else (float(short_edge), short_edge / ratio)
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        width, height = width * scale, height * scale
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,width,height,avg_frame_rate,sample_rate',
                          '-of', 'json', path], capture_output=True, check=True, text=True).stdout
    return json.loads(out)['streams']


def video_reference(path, num_frames):
    """_normalize_video_condition + the soundtrack: uint8 [F][H][W][3] at 24 fps and f32 [2][N] at 32 kHz (or None)."""
    streams = probe(path)
    v = next(s for s in streams if s['codec_type'] == 'video')
    w, h = int(v['width']), int(v['height'])
    num, den = (int(x) for x in v['avg_frame_rate'].split('/'))
    fps = num / den
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if fps != 24:
        scale = 24 / fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
        frames = np.repeat(frames, np.diff(slots, append=int(np.floor(frames.shape[0] * scale + 0.5))), axis=0)
    frames = frames[:num_frames]
    ch, cw = resolve_canvas(w, h)
    if (ch, cw) != (h, w):
        frames = np.stack([np.asarray(Image.fromarray(f).resize((cw, ch), Image.Resampling.LANCZOS)) for f in frames])
    audio = None
    a = [s for s in streams if s['codec_type'] == 'audio']
    if a:
        assert int(a[0]['sample_rate']) == 32000, 'shortcut: soundtracks must be 32 kHz (no resampler in the front end)'
        pcm = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:a:0', '-f', 'f32le', '-ac', '2', '-'],
                             capture_output=True, check=True).stdout
        audio = np.frombuffer(pcm, np.float32).reshape(-1, 2).T[:, :int(num_frames / 24 * 32000)]
    return np.ascontiguousarray(frames), audio


def video_blocks(frames, sample_fps=2.0, fps=24.0):
    """_sample_video_condition_frames + Qwen3VLVideoProcessor: per vision block f32 [gh * gw][1536], grid, label time."""
    stride, idx, cursor = fps / sample_fps, [], 0.0
    while round(cursor) < frames.shape[0]:
        if not idx or round(cursor) > idx[-1]:
            idx.append(round(cursor))
        cursor += stride
    ts = [i / sample_fps for i in range(len(idx))]
    ts += [ts[-1]] * (-len(ts) % 2)
    labels = [(ts[i] + ts[i + 1]) / 2 for i in range(0, len(ts), 2)]
    sel = frames[idx]
    n, h, w = sel.shape[:3]
    t_bar = round(n / 2) * 2
    hb, wb = round(h / 32) * 32, round(w / 32) * 32
    assert 4096 <= t_bar * hb * wb <= 25165824 and (hb, wb) == (h, w), 'shortcut: video frames must not need the processor resize'
    x = (sel.astype(np.float32).transpose(0, 3, 1, 2) / 255.0 - 0.5) / 0.5
    if n % 2:
        x = np.concatenate([x, x[-1:]])
    gh, gw = h // 16, w // 16
    p = x.reshape(-1, 2, 3, gh // 2, 2, 16, gw // 2, 2, 16).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    p = np.ascontiguousarray(p.reshape(-1, gh * gw, 1536), np.float32)
    return [p[i] for i in range(p.shape[0])], (gh, gw), labels


def match_size(img, width, height):
    w, h = img.size
    s = min(1.0, ((width * height) / (w * h)) ** 0.5)
    size = (max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32))
    return img.convert('RGB').resize(size, Image.Resampling.LANCZOS)


def smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=16777216):
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = np.sqrt(h * w / max_pixels)
        hb, wb = max(factor, int(np.floor(h / beta / factor)) * factor), max(factor, int(np.floor(w / beta / factor)) * factor)
    elif hb * wb < min_pixels:
        beta = np.sqrt(min_pixels / (h * w))
        hb, wb = int(np.ceil(h * beta / factor)) * factor, int(np.ceil(w * beta / factor)) * factor
    return hb, wb


def vision_pixels(img):
    """Qwen2VLImageProcessorFast for one image: [gh * gw][3 * 2 * 16 * 16] f32 and (gh, gw) in 16 px patches."""
    w, h = img.size
    hb, wb = smart_resize(h, w)
    if (hb, wb) != (h, w):   # shortcut: PIL bicubic, not torchvision's; match-mode references are already multiples of 32
        img = img.resize((wb, hb), Image.Resampling.BICUBIC)
    x = (np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    x = np.stack([x, x])                                          # temporal patch 2: the image repeated
    gh, gw = hb // 16, wb // 16
    p = x.reshape(1, 2, 3, gh // 2, 2, 16, gw // 2, 2, 16)        # grid_t, tp, c, h/2, merge, p, w/2, merge, p
    p = p.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)                    # t, h/2, w/2, mh, mw, c, tp, ph, pw
    return np.ascontiguousarray(p.reshape(gh * gw, 3 * 2 * 16 * 16), np.float32), (gh, gw)


def sigmas(steps, shift):
    base = np.linspace(1.0, 0.0, steps, dtype=np.float32)        # torch.linspace in fp32
    s = (np.float32(shift) * base / (np.float32(1) + np.float32(shift - 1) * base)).astype(np.float32)
    return [float(v) for i, v in enumerate(s) if i == 0 or v != s[i - 1]]


def spatial_grid(dim, sqrt_area):
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // 2, endpoint=False) * 32


def frame_grid(h, w):
    sq = np.sqrt(h * w)
    hg, wg = spatial_grid(h, sq), spatial_grid(w, sq)
    yy, xx = np.meshgrid(hg, wg, indexing='ij')
    return np.stack([yy.ravel(), xx.ravel()], -1), wg


def temporal_grid(n, origin):
    spans = np.array([5.0 / 3.0 * (1, 4, 4, 4, 4)[i % 5] for i in range(n)], np.float64)
    return origin + np.concatenate([np.zeros(1), np.cumsum(spans[:-1])])


def audio_positions(pos, rows, n, clock, width_grid):
    pos[rows, 0] = np.tile(clock + np.arange(n, dtype=np.float64), 2)
    pos[rows, 2] = np.concatenate([np.full(n, width_grid[0]), np.full(n, width_grid[-1])])


def layout(text_tags, blocks, lt, lh, lw, na):
    """build_ref2va_packed_sequence: [text | reference blocks | target audio | target video]. blocks: (kind, latent h,
    latent w, latent frames, soundtrack latents) per visual reference in packed order."""
    T = len(text_tags)
    nref = sum(f * (h // 2) * (w // 2) for _, h, w, f, _ in blocks)
    naref = sum(2 * n for *_, n in blocks)
    ntv, nta = lt * (lh // 2) * (lw // 2), na * 2
    S = T + nref + naref + nta + ntv
    pos = np.zeros((S, 3), np.float64)
    pos[:T, 0] = np.arange(T, dtype=np.float64)
    tgrid, twidth = frame_grid(lh, lw)
    vidx, aidx, cursor, clock = [], [], T, float(T)
    for kind, h, w, f, n in blocks:
        g, wg = frame_grid(h, w)
        if kind == 'image':
            rows = g.shape[0]
            pos[cursor:cursor + rows, 0] = clock
            pos[cursor:cursor + rows, 1:] = g
            vidx.append(np.arange(cursor, cursor + rows))
            cursor += rows
            clock += 1.0
            continue
        audio_positions(pos, slice(cursor, cursor + 2 * n), n, clock, wg)
        aidx.append(np.arange(cursor, cursor + 2 * n))
        cursor += 2 * n
        rows = f * g.shape[0]
        pos[cursor:cursor + rows, 0] = np.repeat(temporal_grid(f, clock), g.shape[0])
        pos[cursor:cursor + rows, 1:] = np.tile(g, (f, 1))
        vidx.append(np.arange(cursor, cursor + rows))
        cursor += rows
        span = 0.0
        for i in range(f):   # summed sequentially in float64, as the reference
            span += 5.0 / 3.0 * (1, 4, 4, 4, 4)[i % 5]
        clock += max(float(n), span)
    a0, v0 = cursor, cursor + nta
    audio_positions(pos, slice(a0, v0), na, clock, twidth)
    pos[v0:, 0] = np.repeat(temporal_grid(lt, clock), tgrid.shape[0])
    pos[v0:, 1:] = np.tile(tgrid, (lt, 1))
    vidx = np.concatenate(vidx + [np.arange(v0, S)])
    aidx = np.concatenate(aidx + [np.arange(a0, v0)])
    tags = np.empty(S, np.int32)
    tags[:T] = text_tags
    tags[aidx] = 2
    tags[vidx] = 0
    return pos, tags, vidx.astype(np.int32), aidx.astype(np.int32), nref, naref


def prepare(model, req, out, preset='turbo4', seed=0, request_dir='.'):
    """Writes the runner input files of one request into `out` and returns the runner's request line."""
    a = argparse.Namespace(model=model, out=out, preset=preset, seed=seed, request=os.path.join(request_dir, 'request.json'))
    req = req.get('case', req)
    os.makedirs(a.out, exist_ok=True)
    W, H, F = req['width'], req['height'], req['num_frames']
    assert W % 32 == 0 and H % 32 == 0 and F % 17 == 5, 'canvas must be multiples of 32 and frames 17 n + 5'
    lt, lh, lw = (F - 5) // 17 * 5 + 2, H // 16, W // 16
    na = int(round(F / 24 * 40))
    tok = Tokenizer.from_file(os.path.join(a.model, 'tokenizer', 'tokenizer.json'))
    text = lambda s: tok.encode(s, add_special_tokens=False).ids   # noqa: E731
    ids, tags, vis, grids, pxs, dims, auds, blocks = [], [], [], [], [], [], [], []
    ref_dir = os.path.dirname(os.path.abspath(a.request))
    n_img = n_vid = n_aud = 0
    for i, r in enumerate(req['refs']):
        p = r if os.path.isabs(r) else os.path.join(ref_dir, r)
        if not os.path.exists(p):   # R0 kit layout: refs live next to cases.json
            p = os.path.join('/data/paiton/mmh3v/goldens-kit/refs', os.path.basename(r))
        if p.lower().endswith(VIDEO_EXT):
            frames, audio = video_reference(p, F)
            if audio is not None:
                n_aud += 1
                t = text(f'<Audio {n_aud}>: ')
                ids += t
                tags += [1] * len(t)
                audio.astype(np.float32).tofile(f'{a.out}/aud{i}.f32')
                auds.append(f'{a.out}/aud{i}.f32')
            n_vid += 1
            t = text(f'<Video {n_vid}>: ')
            ids += t
            tags += [1] * len(t)
            vb, (gh, gw), labels = video_blocks(frames)
            for j, (pix, lab) in enumerate(zip(vb, labels)):
                t = text(f'<{lab:.1f} seconds>')
                nv = gh * gw // 4
                ids += t + [VISION_START] + [IMAGE_PAD] * nv + [VISION_END]
                tags += [1] * len(t) + [0] * (nv + 2)
                pix.tofile(f'{a.out}/vis{i}_{j}.bin')
                vis.append(f'{a.out}/vis{i}_{j}.bin')
                grids.append(f'{gh}x{gw}')
            fe = max(1, (frames.shape[0] - 5) // 17) * 17 + 5   # the encoder takes 17 n + 5 frames (snap down)
            frames = frames[:fe]
            blocks.append(('video', frames.shape[1] // 16, frames.shape[2] // 16, (fe - 5) // 17 * 5 + 2,
                           0 if audio is None else int(np.ceil(audio.shape[1] / 800))))
        else:
            img = Image.open(p)
            img = match_size(img, W, H) if req.get('ref_mode', 'match') == 'match' else img.convert('RGB')
            pix, (gh, gw) = vision_pixels(img)
            n_img += 1
            t = text(f'<Picture {n_img}>: ')
            nv = gh * gw // 4
            ids += t + [VISION_START] + [IMAGE_PAD] * nv + [VISION_END]
            tags += [1] * len(t) + [0] * (nv + 2)
            pix.tofile(f'{a.out}/vis{i}.bin')
            vis.append(f'{a.out}/vis{i}.bin')
            grids.append(f'{gh}x{gw}')
            frames = np.asarray(img)[None]
            blocks.append(('image', img.size[1] // 16, img.size[0] // 16, 1, 0))
        np.ascontiguousarray(frames).tofile(f'{a.out}/ref{i}.u8')
        pxs.append(f'{a.out}/ref{i}.u8')
        dims.append(f'{frames.shape[0]}x{frames.shape[1]}x{frames.shape[2]}')
    t = text(req['prompt'])
    ids += t
    tags += [1] * len(t)
    pos, stags, vidx, aidx, nref, naref = layout(np.array(tags, np.int32), blocks, lt, lh, lw, na)
    np.array(ids, np.int32).tofile(a.out + '/ids.bin')
    stags.tofile(a.out + '/tags.bin')
    pos.tofile(a.out + '/pos.bin')
    vidx.tofile(a.out + '/vidx.bin')
    aidx.tofile(a.out + '/aidx.bin')
    steps = PRESETS[a.preset]
    sv, sa = sigmas(steps, SHIFT_VIDEO), sigmas(steps, SHIFT_AUDIO)
    line = (f'--ids {a.out}/ids.bin --tags {a.out}/tags.bin --pos {a.out}/pos.bin --vidx {a.out}/vidx.bin --aidx {a.out}/aidx.bin '
            f'--cond-video {nref} --cond-audio {naref} --sigmas {",".join(repr(s) for s in sv)} --audio-sigmas {",".join(repr(s) for s in sa)} '
            f'--latent {lt}x{lh}x{lw} --seed {a.seed}')
    if vis:
        line += f' --vis {",".join(vis)} --vis-grid {",".join(grids)} --ref-px {",".join(pxs)} --ref-dims {",".join(dims)}'
    if auds:
        line += f' --ref-audio {",".join(auds)}'
    open(a.out + '/request.line', 'w').write(line + '\n')
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--request', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--preset', default='turbo4')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(prepare(a.model, json.load(open(a.request)), a.out, a.preset, a.seed, os.path.dirname(os.path.abspath(a.request))))


if __name__ == '__main__':
    main()
