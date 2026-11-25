"""Pinned Inkling 2 Mini server for one R9700: verified checkpoint, then vLLM on Paiton's native libraries."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / 'checkpoint.lock.json').read_text())


def prepare(offline=False, verify=False, lock=LOCK):
    """Download (unless offline) and check the files of a lock entry: the checkpoint, or a drafter of its own."""
    from huggingface_hub import snapshot_download
    started = time.perf_counter()
    snapshot = Path(snapshot_download(
        lock['model'], revision=lock['revision'],
        allow_patterns=[item['name'] for item in lock['files']],
        max_workers=2, local_files_only=offline))
    downloaded = time.perf_counter()
    receipt = Path(os.environ.get('XDG_CACHE_HOME', '/models/cache')) / 'verified' / (lock['revision'] + '.json')
    try:
        previous = json.loads(receipt.read_text())
    except (OSError, ValueError):
        previous = {}
    current = {}
    for item in lock['files']:
        path = snapshot / item['name']
        stat = path.stat()
        if stat.st_size != item['bytes']:
            raise ValueError('Checkpoint size mismatch: ' + item['name'])
        signature = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                     stat.st_dev, stat.st_ino, item['sha256']]
        if verify or previous.get(item['name']) != signature:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != item['sha256']:
                raise ValueError('Checkpoint SHA256 mismatch: ' + item['name'])
        current[item['name']] = signature
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(current))
    temporary.replace(receipt)
    print(json.dumps(dict(model=lock['model'], revision=lock['revision'],
        preparation_seconds=downloaded-started,
        verification_seconds=time.perf_counter()-downloaded)), flush=True)
    return snapshot


# The model's own drafter (DSpark, mtp.safetensors): 5 draft tokens while one request runs, 4 from two on
# (measured on image c4: 5 beats 7 at temperature 0 and at the model's default 1.0; 7 stays the upper bound).
DRAFTER = {'method': 'dspark', 'num_speculative_tokens': 7,
           'num_speculative_tokens_per_batch_size': [[1, 1, 5], [2, 4, 4]]}
# --drafter dflash2: a DFlash2 draft checkpoint of its own (checkpoint.lock.json "drafters": a third-party repository,
# fetched at its pinned revision into the same cache as the checkpoint; NOTICE.md). It always drafts a block of 7;
# the table says how many of them the server checks. Served through the same vLLM method as the model's own drafter.
DFLASH2 = {'method': 'dspark', 'num_speculative_tokens': 7,
           'num_speculative_tokens_per_batch_size': [[1, 1, 5], [2, 4, 4]]}
DEFAULT_DRAFTER = 'dspark'
DFLASH2_SO = 'PAITON_INK2_DFLASH2_SO'  # the image's DFlash2 drafter library

# 16 GB cards (RX 9070 / 9070 XT, RX 9060 XT 16 GB). Model and drafter take about 20 GiB, so the routed experts the
# card cannot hold stay in pinned host memory and a miss is fetched, never skipped: the output stays exact.
SMALL_CARD_GIB = 20.0    # free VRAM below this at start selects the profile
SMALL_CONTEXT = 32768    # its default context; --max-model-len 65536 for 64K (a smaller expert pool, more misses)
SMALL_FIXED_GIB = 4.67   # resident besides experts and cache, drafter on: decoder, head rows, drafter, runtime, KFD headroom
DRAFTER_GIB, DFLASH2_GIB, VISION_GIB = 0.69, 0.92, 0.59
EXPERTS_GIB, EMBED_GIB = 14.59, 0.86  # every expert keeps a host copy for the pool to fetch from; the embedding moves too
SERVER_HOST_GIB = 8.0    # the server's own unpinned host memory while it loads
# Pool hit rate by GiB of experts in VRAM, drafter on and off (20 real-text requests of image c10 replayed through
# the native pool, compiler native/inkling2_pool ABI 2, two pool groups by expert format, 20 % of the bytes MXFP8).
HIT = {True: ((5.98, 0.888), (8.26, 0.952), (9.39, 0.970)), False: ((7.29, 0.948), (9.29, 0.975), (10.29, 0.983))}
MEMINFO = Path('/proc/meminfo')
EXPERT_POOL = False  # shortcut: this image has no expert pool yet (16 GB plan K1 to K3, N3); True once it ships


def amd_cards():
    """sysfs folders of the AMD GPUs on this host."""
    for dev in Path('/sys/class/drm').glob('renderD*/device'):
        try:
            if (dev / 'vendor').read_text().strip() == '0x1002':
                yield dev
        except OSError:
            continue


def card_free_gib(dri=Path('/dev/dri')):
    """Free VRAM of the smallest AMD card this process can open (a container has only its own render nodes).
    Integrated GPUs (under 8 GiB of VRAM, such as the one in Ryzen 7000/9000 desktop CPUs) do not count."""
    free = []
    for dev in amd_cards():
        total = int((dev / 'mem_info_vram_total').read_text())
        if (dri / dev.parent.name).exists() and total >= 8 * 2 ** 30:
            free.append(total - int((dev / 'mem_info_vram_used').read_text()))
    return min(free) / 2 ** 30 if free else None


def small_card(free, context, drafter, vision, kv, need_pool=True):
    """(vLLM arguments, environment, plan) of the 16 GB profile; stops with the reason when it cannot run here."""
    from paiton_vllm_plugin.inkling2.cache_plan import cache_bytes
    # drafter: False, True or 'dspark' (the model's own), or 'dflash2'
    dflash2, drafter = drafter == 'dflash2', bool(drafter)
    cache = cache_bytes(context, 1, 256, kv=kv, drafter='dflash2' if dflash2 else drafter) / 2 ** 30
    rope = context * 512 / 2 ** 30 if drafter else 0.0
    pool = (free - SMALL_FIXED_GIB + (0 if drafter else DRAFTER_GIB) - (DFLASH2_GIB - DRAFTER_GIB if dflash2 else 0)
            - (VISION_GIB if vision else 0) - cache - rope)
    points = HIT[drafter]  # shortcut: the hit rates were replayed with the model's own drafter; replay with DFlash2 before the 16 GB build ships it
    hit = next((f'{a[1] + (b[1] - a[1]) * (pool - a[0]) / (b[0] - a[0]):.2f}' for a, b in zip(points, points[1:])
                if a[0] <= pool <= b[0]), f'{points[-1][1]:.2f} or more' if pool > points[-1][0] else f'under {points[0][1]:.2f}')
    note = (f'16 GB profile: {free:.1f} GiB free VRAM; cache {cache:.2f} GiB for {context} tokens, one request; '
            f'{pool:.2f} of {EXPERTS_GIB} GiB of experts on the card (replay hit rate {hit}), the rest in host memory')
    if need_pool and not EXPERT_POOL:
        raise SystemExit(f'{note}. This image keeps every expert on the card and needs about 20 GiB free; '
                         'the 16 GB build is not released yet')
    if pool < 1.0:
        raise SystemExit(f'{note}: too few experts on the card; free the card or serve a shorter --max-model-len')
    meminfo = dict(line.split(':', 1) for line in MEMINFO.read_text().splitlines())
    avail = int(meminfo['MemAvailable'].split()[0]) / 2 ** 20 - SERVER_HOST_GIB
    totals = [int((dev / 'mem_info_gtt_total').read_text()) for dev in amd_cards()]
    gtt = (min(totals) - sum(int((dev / 'mem_info_gtt_used').read_text()) for dev in amd_cards())) / 2 ** 30
    if min(avail, gtt) < EXPERTS_GIB + EMBED_GIB:
        raise SystemExit(f'{note}. The expert and embedding copies need {EXPERTS_GIB + EMBED_GIB:.2f} GiB of pinned host '
                         f'memory; this host has {avail:.1f} GiB available (less {SERVER_HOST_GIB:g} for the server) '
                         f'and {gtt:.1f} GiB of GTT room')
    args = ['--max-model-len', str(context), '--max-num-seqs', '1', '--max-num-batched-tokens', '256',
            '--kv-cache-memory-bytes', str(int(cache * 2 ** 30))]
    if not vision:
        args += ['--limit-mm-per-prompt', json.dumps({'image': 0})]
    env = dict(HSA_USERPTR_FOR_PAGED_MEM='0', PAITON_INK2_HOST_EMBED='1', PAITON_INK2_EXPERT_POOL_GIB=f'{pool:.2f}')
    return args, env, note


def command(snapshot, host='0.0.0.0', extra=(), drafter=True, draft_snapshot=None):
    # drafter: False, True or 'dspark' (the model's own), or 'dflash2' with the folder of its checkpoint
    config = {**DFLASH2, 'model': str(draft_snapshot)} if drafter == 'dflash2' else DRAFTER
    # Later arguments win in vLLM's parser, so `extra` (what the user adds after the image name) overrides these.
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(snapshot), '--served-model-name', os.environ.get('PAITON_SERVED_MODEL', 'inkling-2-mini'),
        '--host', host, '--port', '8042', '--quantization', 'inkling2', '--enforce-eager',
        '--max-model-len', MAX_MODEL_LEN, '--max-num-seqs', '4', '--max-num-batched-tokens', '512',
        '--reasoning-parser', 'inkling2', '--enable-auto-tool-choice', '--tool-call-parser', 'inkling2',
        *(['--speculative-config', json.dumps(config)] if drafter else []), *extra]


# auto: vLLM sizes the context to the cache memory the card has left (322,736 tokens on a 32 GB R9700).
MAX_MODEL_LEN = 'auto'
# The release's cache format of the 8 full-attention visits; --kv-cache or -e PAITON_INK2_KV picks another.
KV_CACHE = 'bf16'

# Prompt-cache tiers (opt-in: --host-cache-gib, --extend-cache). Blocks the card's prefix cache evicts stay in pinned
# system memory (vLLM's OffloadingConnector) and, with --extend-cache disk, go on to files (its TieringOffloadingSpec
# file tier), so a long conversation keeps its prefix after the card evicts it. Every cache group of this model is
# attention-like (full visits, 1,024-token windows, 4-token conv states as a window), which the connector takes as it
# is; it keeps window and conv chunks only before the semantic checkpoints (prompt end, shared prefix), as the card's own
# cache does. The image carries the 27B release's offload worker: with PAITON_HOST_KV_PRIVATE=1 the RAM tier is one
# exact-size hipHostMalloc allocation per tensor, and a tier that cannot be pinned stops the start instead of falling
# back to pageable copies.
SERVER_PEAK_GIB = 8.0      # the server's own unpinned host memory while it loads
HOST_RESERVE_GIB = 7.0     # the system, a desktop and what must stay available (as the 27B launcher)
GTT_MARGIN_GIB = 0.5
# The card's pool on the R9700 (25 Nov 2026, input embedding on the host): 9.9 GiB with the drafter, 11.2 GiB without.
CARD_POOL_GIB = 12.0       # the card's prefix-cache pool: a RAM tier is inclusive, so it must clearly exceed this
# The restore set of a pool-sized document: pool tokens x tier bytes per token (39.6 KB with the drafter) + ~0.3 GiB of
# window and conv chunks at the boundary; ~10.8 GiB without the drafter.
STAGING_GIB = 12.0         # disk tier: the system-memory stage every restore passes; holds a document the size of the pool
MIN_TIER_GIB = 4.0
DISK_MAX_GIB, DISK_FREE_SHARE = 64.0, 0.25
# vLLM's file tier writes one file per chunk. 16 blocks (256 tokens, 9 MiB files) restore 146K tokens after a restart in
# 2.2 s instead of 9.1 s with one block (576 KiB files), same disk use per token; a hit then ends up to 255 tokens
# before the prompt end instead of 15 (25 Nov 2026, R9700).
DISK_BLOCKS_PER_CHUNK = 16
DISK_ROOT = Path(os.environ.get('XDG_CACHE_HOME', '/models/cache')) / 'kv-disk'


def pin_limit_gib():
    """GiB the tier may pin now (0.5 GiB steps): MemAvailable less the server's own peak and a reserve, at most the GTT
    room the AMD GPUs have left. PAITON_HOST_PIN_LIMIT_GIB overrides it."""
    if os.environ.get('PAITON_HOST_PIN_LIMIT_GIB'):
        return float(os.environ['PAITON_HOST_PIN_LIMIT_GIB'])
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    room = int(meminfo['MemAvailable'].split()[0]) / 2 ** 20 - SERVER_PEAK_GIB - HOST_RESERVE_GIB
    sizes, used = [], 0
    for device in Path('/sys/class/drm').glob('renderD*/device'):
        try:
            if (device / 'vendor').read_text().strip() == '0x1002':
                sizes.append(int((device / 'mem_info_gtt_total').read_text()))
                used += int((device / 'mem_info_gtt_used').read_text())
        except (OSError, ValueError):
            continue
    if sizes:  # one TTM limit for every GPU: the smallest GTT size less what all of them use
        room = min(room, (min(sizes) - used) / 2 ** 30 - GTT_MARGIN_GIB)
    return max(0.0, math.floor(2 * room) / 2)


def disk_folder():
    """The disk tier's folder for this configuration. vLLM's own folder key (model, dtype, cache groups) does not change
    with the projection format (--int8-attention) or a new library build, which change the cached values."""
    manifest = HERE / 'lib' / 'MANIFEST.json'
    key = json.dumps([sorted((k, v) for k, v in os.environ.items() if k.startswith('PAITON_INK2')),
                      manifest.read_text() if manifest.is_file() else None])
    return DISK_ROOT / hashlib.sha256(key.encode()).hexdigest()[:16]


def folder_bytes(folder):
    return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file()) if folder.is_dir() else 0


def tier_settings(host_cache_gib, extend_cache, pinned_gib=0.0):
    """(vLLM arguments, note) of the prompt-cache tiers; sets PAITON_HOST_KV_PRIVATE for the RAM tier. `pinned_gib`: what
    the server pins itself (embedding, the 16 GB profile's expert copies), taken off the limit unless it is overridden."""
    if host_cache_gib and extend_cache:
        raise SystemExit('--extend-cache sizes the tier itself: drop --host-cache-gib or --extend-cache')
    limit = pin_limit_gib()
    if not os.environ.get('PAITON_HOST_PIN_LIMIT_GIB'):
        limit = max(0.0, math.floor(2 * (limit - pinned_gib)) / 2)
    choice = extend_cache
    if host_cache_gib:
        if host_cache_gib > limit:
            raise SystemExit(f'--host-cache-gib {host_cache_gib:g}: this host can pin {limit:g} GiB now (MemAvailable '
                             f'less {SERVER_PEAK_GIB + HOST_RESERVE_GIB:g} GiB and the {pinned_gib:.2f} GiB the server '
                             f'pins itself, at most the free GTT less '
                             f'{GTT_MARGIN_GIB:g} GiB; PAITON_HOST_PIN_LIMIT_GIB overrides it)')
        choice = 'ram'
    elif choice == 'auto':
        choice = 'ram' if limit >= 1.25 * CARD_POOL_GIB else 'disk'
    ram = host_cache_gib or (limit if choice == 'ram' else min(limit, STAGING_GIB))
    if ram < MIN_TIER_GIB:
        raise SystemExit(f'--extend-cache: this host can pin {limit:g} GiB now, under the {MIN_TIER_GIB:g} GiB a tier needs')
    args = ['--kv-offloading-size', f'{ram:g}', '--kv-offloading-backend', 'native']
    if choice == 'ram':
        os.environ['PAITON_HOST_KV_PRIVATE'] = '1'
        return args, (f'prompt-cache tier: {ram:g} GiB of system memory (--extend-cache {extend_cache}: largest this '
                      'host can pin now)' if extend_cache else f'prompt-cache tier: {ram:g} GiB of system memory')
    # Disk: the stage is a /dev/shm file (the tiering spec maps one), which a container needs room for.
    shm = shutil.disk_usage('/dev/shm').total / 2 ** 30
    if shm < ram + 0.5:
        raise SystemExit(f'--extend-cache disk stages {ram:g} GiB in /dev/shm, which holds {shm:.1f} GiB here: run the '
                         f'container with --ipc=private --shm-size={math.ceil(STAGING_GIB) + 1}g (serve-docker.sh does)')
    folder = disk_folder()
    DISK_ROOT.mkdir(parents=True, exist_ok=True)
    cap = min(DISK_MAX_GIB, DISK_FREE_SHARE * (shutil.disk_usage(DISK_ROOT).free / 2 ** 30
                                                + folder_bytes(DISK_ROOT) / 2 ** 30))
    if cap < 1.25 * CARD_POOL_GIB:
        raise SystemExit(f'--extend-cache disk: a quarter of the free space under {DISK_ROOT} is {cap:.0f} GiB, under '
                         f'{1.25 * CARD_POOL_GIB:g} GiB: free some space')
    # The file tier has no limit of its own while it runs: an over-full folder is removed here, the other
    # configurations' folders (never hit by this one) first.
    for other in sorted(DISK_ROOT.iterdir(), key=lambda p: p == folder):
        if folder_bytes(DISK_ROOT) / 2 ** 30 <= cap:
            break
        shutil.rmtree(other)
    # Files must match after a restart: a cryptographic block hash with vLLM's fixed seed.
    args += ['--prefix-caching-hash-algo', 'sha256', '--kv-transfer-config', json.dumps({
        'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both',
        'kv_connector_extra_config': {'spec_name': 'TieringOffloadingSpec', 'blocks_per_chunk': DISK_BLOCKS_PER_CHUNK,
                                      'secondary_tiers': [{'type': 'fs', 'root_dir': str(folder)}]}})]
    return args, (f'prompt-cache tier: files under {folder} (up to {cap:.0f} GiB, checked at start) behind a {ram:g} GiB '
                  'system-memory stage')


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Other arguments go to vLLM unchanged.')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Rehash all checkpoint files, ignoring verification receipts')
    parser.add_argument('--host', default='0.0.0.0',
                        help='address the API listens on (default 0.0.0.0, every interface; 127.0.0.1 keeps it local)')
    parser.add_argument('--api-key', default=None,
                        help='opt-in: require this key on /v1 requests (passed to vLLM as VLLM_API_KEY, never printed)')
    parser.add_argument('--max-model-len', type=int, default=None,
                        help='context in tokens (default: what the cache holds on a 32 GB card, 32,768 on a 16 GB card)')
    parser.add_argument('--dry-run', nargs='?', type=float, const=-1.0, default=None, metavar='FREE_GIB',
                        help='print the plan, the environment and the vLLM command without downloading or serving; '
                             'FREE_GIB plans for a card with that much free VRAM (15.9 for a 16 GB card)')
    parser.add_argument('--vision', action='store_true',
                        help='16 GB cards: load the vision tower (off there by default; 0.59 GiB less for experts)')
    parser.add_argument('--int8-attention', action='store_true',
                        help='opt-in: attention projection weights as 8-bit rows instead of bf16; about 9 %% faster '
                             'decode, a small measurable change of the output distribution (KL +0.003)')
    parser.add_argument('--kv-cache', choices=('bf16', 'fp8', 'kv4'), default=None,
                        help='cache format of the 8 full-attention visits: bf16 32,768 bytes per context token, fp8 '
                             f'half, kv4 about a quarter (default {KV_CACHE}); the window visits stay bf16')
    parser.add_argument('--drafter', choices=('dspark', 'dflash2'), default=DEFAULT_DRAFTER,
                        help='dspark: the drafter in the model checkpoint. dflash2: the DFlash2 drafter '
                             f'{LOCK["drafters"]["dflash2"]["model"]} (Apache-2.0, a third-party checkpoint), '
                             'downloaded at first start (1.0 GB) and checked like the model. Default: %(default)s')
    parser.add_argument('--no-drafter', action='store_true',
                        help='serve without the drafter: one token per step, 0.7 GiB more memory for the cache')
    parser.add_argument('--draft-cut', type=float, metavar='P', default=None,
                        help='opt-in: verify only the drafts whose running probability stays above P (0.4 measured '
                             '6 %% faster for one request); the server then runs without async scheduling')
    parser.add_argument('--host-cache-gib', type=float, metavar='GIB', default=None,
                        help='opt-in: keep prompt-cache blocks the card evicts in GIB of pinned system memory, so a '
                             'long prompt sent again loads instead of being computed again')
    parser.add_argument('--extend-cache', nargs='?', const='auto', choices=('auto', 'ram', 'disk'), default=None,
                        help='opt-in: size that tier from this host: ram = all the memory it can pin now, disk = files '
                             f'under the cache folder (NVMe/SSD; also across restarts) behind a {STAGING_GIB:g} GiB stage; '
                             f'auto (the default) = ram when that holds {1.25 * CARD_POOL_GIB:g} GiB or more')
    args, extra = parser.parse_known_args()
    if args.host_cache_gib is not None and not 0 < args.host_cache_gib < math.inf:
        parser.error('--host-cache-gib must be a number of GiB greater than 0')
    from paiton_vllm_plugin.activation import activate_model_launcher
    activate_model_launcher(False, os.environ)
    drafter = False if args.no_drafter else args.drafter
    planned = args.dry_run is not None and args.dry_run >= 0
    free = None if args.download_only else args.dry_run if planned else card_free_gib()
    if free is not None and free < SMALL_CARD_GIB:  # before the download: a card that cannot serve stops here
        small, env, note = small_card(free, args.max_model_len or SMALL_CONTEXT, drafter, args.vision,
                                      args.kv_cache or 'bf16', need_pool=args.dry_run is None)
        print(json.dumps({'card_profile': note}), flush=True)
        for name, value in env.items():
            os.environ.setdefault(name, value)  # `docker run -e` wins
        extra = [*small, *extra]
    elif args.max_model_len:
        extra = ['--max-model-len', str(args.max_model_len), *extra]
    snapshot = '<checkpoint>' if args.dry_run is not None else prepare(args.offline, args.verify)
    draft_snapshot = None
    if drafter == 'dflash2':
        draft_snapshot = ('<DFlash2 drafter>' if args.dry_run is not None
                          else prepare(args.offline, args.verify, LOCK['drafters']['dflash2']))
    if args.download_only:
        return
    if drafter == 'dflash2' and args.dry_run is None and not os.path.exists(os.environ.get(DFLASH2_SO, '')):
        raise SystemExit(f'--drafter dflash2: this image has no DFlash2 drafter library ({DFLASH2_SO})')
    os.environ['HF_HUB_OFFLINE'] = '1'
    if args.draft_cut is not None:
        os.environ['PAITON_INK2_DRAFT_CUT'] = str(args.draft_cut)
        extra = ['--no-async-scheduling', *extra]
    if args.int8_attention:
        os.environ['PAITON_INK2_PROJ'] = 'int8'
    if args.kv_cache:
        os.environ['PAITON_INK2_KV'] = args.kv_cache
    # Defaults of the release; `docker run -e NAME=value` overrides one for a comparison. PAITON_INK2_HOST_EMBED=1: the
    # input embedding in 0.86 GiB of pinned host memory (exact; about 24K more context); -e PAITON_INK2_HOST_EMBED=0
    # keeps it on the card.
    for name, value in dict(PAITON_INKLING2='1', PAITON_INK2_EXPERTS='native', PAITON_INK2_OPS='native',
                            PAITON_INK2_STEP='native', PAITON_INK2_STEP_FLAGS='7', PAITON_INK2_LMHEAD='int8',
                            PAITON_INK2_DRAFT_LOGITS='int8', PAITON_INK2_KV=KV_CACHE,
                            PAITON_INK2_HOST_EMBED='1').items():
        os.environ.setdefault(name, value)
    if args.host_cache_gib or args.extend_cache:
        pinned = ((EMBED_GIB if os.environ['PAITON_INK2_HOST_EMBED'] == '1' else 0.0)
                  + (EXPERTS_GIB if os.environ.get('PAITON_INK2_EXPERT_POOL_GIB') else 0.0))
        tier, note = tier_settings(args.host_cache_gib, args.extend_cache, pinned)  # after the PAITON_INK2 settings: disk key
        print(json.dumps({'cache_tier': note}), flush=True)
        extra = [*tier, *extra]
    if args.api_key:
        os.environ['VLLM_API_KEY'] = args.api_key
    argv = command(snapshot, args.host, extra, drafter, draft_snapshot)
    print(json.dumps({'command': argv}), flush=True)
    if args.dry_run is not None:
        print(json.dumps({'environment': {k: v for k, v in sorted(os.environ.items()) if k.startswith(('PAITON_', 'HSA_'))}}))
        return
    os.execv(sys.executable, argv)


if __name__ == '__main__':
    main()
