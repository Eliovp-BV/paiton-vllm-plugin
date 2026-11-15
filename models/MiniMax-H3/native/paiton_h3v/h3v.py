"""MiniMax H3 Ref2VA on one R9700: command line around the native runner (h3v_run). Host side only: numpy, PIL,
tokenizers, ffmpeg; no torch.

    python3 h3v.py generate --prompt "..." --ref a.png [--ref b.mp4 ...] [--width 864 --height 480 --frames 124]
                            [--seed N] --out clip.mp4
    python3 h3v.py serve        # weights load once; one JSON request per stdin line:
                                # {"prompt": .., "refs": [..], "width": .., "height": .., "num_frames": .., "seed": .., "out": ".."}
                                # answered with "DONE <path>" or "ERROR <message>"

H3V_HOME (or --home) holds what the runner loads:
  weights/<preset>/        dit.bin, dit.idx, vae_latents.bin (convert.py; one bundle per preset: turbo4, turbo8)
  libs/                    h3v_run and the component libraries (build_runner.sh)
  te-w8/ [te-w4/]          K4 text encoder directories (W8A8; the calibrated W4 for hosts under 48 GiB of RAM)
  dec8.bin enc.bin audio.bin   V2 int8 decoder, V3 encoder and audio VAE blobs
  model/                   the diffusers MiniMax-H3 folder (tokenizer only is read)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import front  # noqa: E402


def runner_cmd(home, preset, extra):
    mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    te = os.path.join(home, 'te-w4' if mem < 48 * 2**30 and os.path.isdir(os.path.join(home, 'te-w4')) else 'te-w8')
    return [os.path.join(home, 'libs', 'h3v_run'), '--weights', os.path.join(home, 'weights', preset), '--libs', os.path.join(home, 'libs'),
            '--te', te, '--vae', os.path.join(home, 'dec8.bin'), '--enc', os.path.join(home, 'enc.bin'),
            '--audio', os.path.join(home, 'audio.bin'), '--max-tokens', '8192'] + extra


def request_line(home, preset, req, work):
    line = front.prepare(os.path.join(home, 'model'), req, work, preset, int(req.get('seed', 0)), os.getcwd())
    out = os.path.abspath(req['out'])
    return f"{line} --out {out[:-4] if out.endswith('.mp4') else out}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('generate', 'serve'))
    ap.add_argument('--home', default=os.environ.get('H3V_HOME', ''))
    ap.add_argument('--preset', default='turbo4', choices=tuple(front.PRESETS))
    ap.add_argument('--prompt')
    ap.add_argument('--ref', action='append', default=[])
    ap.add_argument('--width', type=int, default=864)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--frames', type=int, default=124)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out')
    ap.add_argument('--runner-args', default='', help='extra runner options, e.g. "--te-resident 6 --vram-cap-gib 27.5"')
    a = ap.parse_args()
    if not a.home:
        sys.exit('set H3V_HOME or --home')
    extra = a.runner_args.split()
    if a.cmd == 'generate':
        if not (a.prompt and a.ref and a.out):
            sys.exit('generate needs --prompt, at least one --ref and --out')
        req = dict(prompt=a.prompt, refs=[os.path.abspath(r) for r in a.ref], width=a.width, height=a.height,
                   num_frames=a.frames, seed=a.seed, out=a.out)
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.out))) as work:
            line = request_line(a.home, a.preset, req, work)
            sys.exit(subprocess.run(runner_cmd(a.home, a.preset, extra + line.split())).returncode)
    # serve: the runner keeps the weights; each request gets its own front-end directory
    proc = subprocess.Popen(runner_cmd(a.home, a.preset, extra + ['--serve']), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in proc.stdout:   # load messages until READY
        sys.stderr.write(line)
        if line.startswith('READY'):
            break
    print('READY', flush=True)
    for raw in sys.stdin:
        if not raw.strip():
            continue
        try:
            req = json.loads(raw)
            with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(req['out']))) as work:
                proc.stdin.write(request_line(a.home, a.preset, req, work) + '\n')
                proc.stdin.flush()
                reply = 'ERROR the runner exited'
                for reply in proc.stdout:
                    sys.stderr.write(reply)
                    if reply.startswith(('DONE', 'ERROR')):
                        break
            print(f"DONE {req['out']}" if reply.startswith('DONE') else reply.strip(), flush=True)
        except Exception as e:   # bad request: report and keep serving
            print(f'ERROR {e}', flush=True)
    proc.stdin.close()
    proc.wait()


if __name__ == '__main__':
    main()
