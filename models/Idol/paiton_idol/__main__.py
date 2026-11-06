"""Idol (one image -> 3D Gaussians .ply) with the Paiton native denoiser on a Radeon AI PRO R9700.

  python -m paiton_idol download
  python -m paiton_idol generate IMAGE [IMAGE ...] [--out DIR] [--mode synthetic|real] [--exact] [--seed N]
                                       [--gaussians N] [--steps N] [--cfg X]
  python -m paiton_idol benchmark IMAGE_DIR       (median request time over the images after one warm-up)
"""
import argparse
import glob
import json
import os
import shutil
import statistics
import sys
import time

ROOT = os.environ.get('PAITON_IDOL_HOME', '/models/idol')
FAST_FROM = 7    # default: int8 QK + fp8 PV attention from step 7 of 50 (render-gated), fp16 GEMMs on every step


def load(args):
    from paiton_idol import weights
    run = weights.prepare(ROOT)
    os.environ['HF_HUB_OFFLINE'] = '1'   # everything is local from here on
    # MIOpen picks convolution kernels (BiRefNet, DINOv3, VAE) by timing on a fresh cache, which shifts the inputs to
    # the denoiser; seed the user find-db with the one the release was gated with, so a new install picks the same.
    db = os.environ.get('MIOPEN_USER_DB_PATH')
    if db:
        os.makedirs(db, exist_ok=True)
        src = os.path.join(os.path.dirname(__file__), 'miopen')
        for f in os.listdir(src):
            if not os.path.exists(os.path.join(db, f)):
                shutil.copy(os.path.join(src, f), os.path.join(db, f))
    import torch
    from idol import pipeline as pl
    from paiton_idol.native import NativeSampler

    class NativeDenoiser:   # stands in for the PyTorch denoiser: the native library loads its weights itself
        def __init__(self, q_token_length, in_channels, cam_channels, **_):
            self.q_token_length, self.in_channels, self.cam_channels = q_token_length, in_channels, cam_channels

        def load_state_dict(self, *a, **k):
            pass

        def eval(self):
            return self

        def requires_grad_(self, *_):
            return self

        def to(self, *_):
            return self

    base_load = pl.load_file
    pl.load_file = lambda p, *a, **k: {} if p.endswith('denoiser.safetensors') else base_load(p, *a, **k)
    pl.LatentSeqMMFlowModel = NativeDenoiser
    pl.sample_flow = NativeSampler(os.path.join(run, 'denoiser.safetensors'),
                                   fast_from=None if args.exact else FAST_FROM,
                                   cfg_until=None)
    p = pl.IdolPipeline(run, device='cuda', optimize=False)   # optimize only affects the PyTorch denoiser
    return p, torch


def generate(p, torch, image, args, out_dir):
    name = os.path.splitext(os.path.basename(image))[0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = p(image, mode=args.mode, seed=args.seed, steps=args.steps, cfg_strength=args.cfg, rescale_t=3.0,
            num_gaussians=args.gaussians)
    out.gaussian.save_ply(os.path.join(out_dir, f'{name}.ply'))
    dt = time.perf_counter() - t0
    out.preprocessed_image.save(os.path.join(out_dir, f'{name}_input.png'))
    json.dump(out.camera, open(os.path.join(out_dir, f'{name}_camera.json'), 'w'))
    return name, dt


def main():
    ap = argparse.ArgumentParser(prog='paiton_idol', description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('command', choices=['download', 'generate', 'benchmark'])
    ap.add_argument('inputs', nargs='*')
    ap.add_argument('--out', default='/outputs')
    ap.add_argument('--mode', default='synthetic', choices=['synthetic', 'real'],
                    help='synthetic: renders and generated images; real: photos (wider crop)')
    ap.add_argument('--exact', action='store_true', help='fp16 attention on every step (same output as stock fp16)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gaussians', type=int, default=262144)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--cfg', type=float, default=3.0)
    args = ap.parse_args()
    if args.command == 'download':
        from paiton_idol import weights
        print(weights.prepare(ROOT))
        return
    images = []
    for x in args.inputs:
        images += sorted(glob.glob(os.path.join(x, '*'))) if os.path.isdir(x) else [x]
    images = [x for x in images if x.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not images:
        sys.exit('no input images (.png, .jpg, .jpeg, .webp)')
    os.makedirs(args.out, exist_ok=True)
    p, torch = load(args)
    if args.command == 'benchmark':
        generate(p, torch, images[0], args, args.out)   # warm-up
    times = {}
    for image in images:
        name, dt = generate(p, torch, image, args, args.out)
        times[name] = dt
        print(f'{name}: {dt:.2f} s -> {os.path.join(args.out, name + ".ply")}', flush=True)
    if args.command == 'benchmark':
        r = {'median_request_s': statistics.median(times.values()), 'requests': times, 'steps': args.steps,
             'exact': args.exact, 'gaussians': args.gaussians, 'device': torch.cuda.get_device_name(0)}
        json.dump(r, open(os.path.join(args.out, 'benchmark.json'), 'w'), indent=2)
        print(f'median request time: {r["median_request_s"]:.2f} s over {len(times)} images')


if __name__ == '__main__':
    main()
