#!/usr/bin/env python3
"""Qwen3.8 Flash Next on two AMD R9700 (ROCm 10, vLLM 0.29, tensor parallel 2) with the Paiton plugin image.

Modes (modes.json, from the measured runs on this image):
  decode             default: MTP depth 3, 98,304-token window, KV 1.6 GiB per card              (decode / concurrency numbers)
  prefill-long       200,000-token window, MTP off, 3 GiB KV per card                            (prefill numbers)
  decode-nopf / prefill-long-nopf   the same with exact-arithmetic prefill (about 7 % slower)
  --prefix-caching   opt-in on the 3-bit modes: align-mode prefix caching (byte-identical hits); 4-bit: --mode 4bit-cache
Host pools: 3-bit modes keep ROCm's user-pointer memory for the host-resident tables (~55 GiB, both ranks; the published behaviour);
  --host-pools gtt (opt-in) allocates them as driver-owned GTT memory (HSA_USERPTR_FOR_PAGED_MEM=0): no memory pauses from host memory
  compaction under load, ~1 GiB more VRAM in decode. 4-bit modes use GTT pools when the GPU's GTT size holds them.
  4bit / 4bit-200k   4-bit experts (W4A8 repo) with the expert host tier: experts beyond --expert-vram-gib per GPU live in
                     pinned host memory (about 41 GiB with the default host pool; the launcher checks the RAM first)
  4bit-cache         opt-in for agents and coding tools: 4bit + prefix caching + an 8 GiB RAM prefix tier (needs ~8 GiB more GTT)
  4bit-long / 4bit-xl  4bit with a 262,144-token window (MTP on): KV 3.5 / 4.5 GiB per GPU (~461K / ~595K tokens), experts
                     19.75 / 18.75 GiB per GPU in VRAM; --prefix-caching refused (not measured)
  --ple-host auto|pinned|pageable  4bit / 4bit-cache: the PLE tables (~49 GiB, both ranks) pinned in GTT (as benchmarked) or in
                     pageable host memory (pinned host memory ~96 -> ~47 GiB, so the GTT pools fit a 128 GB host; needs an image with that
                     runtime). auto (default): pageable when the GPU's GTT limit cannot hold the pinned tables (a host with less than
                     ~200 GB of RAM at the default GTT limit), pinned otherwise; the launcher prints which one it chose and why.
Prerequisites: python3 >= 3.10, docker with GPU device access (/dev/kfd, /dev/dri), the huggingface_hub Python package
  (`pip install huggingface_hub`; the `hf` / `huggingface-cli` binaries are used only as a fallback), two AMD GPUs visible.
Usage: launch-flashnext.py [--mode ...] [--prefix-caching] [--port 18982] [--host 0.0.0.0] [--api-key KEY | env PAITON_API_KEY] [--name paiton-flashnext]
       [--detach] [--dry-run] [--ple-host auto|pinned|pageable]
       [--weights DIR] [--cache DIR] [--image REF] [--devices GPU-xxx,GPU-yyy] [--served-model-name Qwen3.8-Flash-Next]
Weights: downloaded once from the Hugging Face repo in modes.json (pinned revision, SHA256SUMS verified) into --weights.
"""
import argparse, json, os, pathlib, re, shlex, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
CFG = json.loads((HERE / 'modes.json').read_text()); COMMON = CFG['common']; MODES = CFG['modes']

SITE_PACKAGES = '/usr/local/lib/python3.12/dist-packages'
# verbatim copy of paiton_vllm_plugin/flashnext/zz_paiton_fn_kv_tier.pth (a no-op unless PAITON_FN_KV_TIER=1)
KV_TIER_PTH = ("import os; os.environ.get('PAITON_FN_KV_TIER', '0') == '1' and os.path.exists('/usr/local/lib/python3.12/dist-packages/paiton_vllm_plugin/flashnext/kv_tier_boot.py') "
               "and __import__('runpy').run_path('/usr/local/lib/python3.12/dist-packages/paiton_vllm_plugin/flashnext/kv_tier_boot.py', run_name='paiton_fn_kv_tier_boot')['install']()")

def amd_gpus():
    """ROCR ids of every AMD GPU (vendor 0x1002) from sysfs; no card numbers or PCI addresses are assumed."""
    out = []
    for dev in sorted(pathlib.Path('/sys/class/drm').glob('card[0-9]*/device')):
        try:
            if (dev / 'vendor').read_text().strip().lower() != '0x1002': continue
            uid = (dev / 'unique_id').read_text().strip().lower().replace('0x', '')
            if uid and uid not in out: out.append(uid)
        except OSError: continue
    return ['GPU-' + u for u in out]

def render_gid():
    try: return subprocess.run(['getent', 'group', 'render'], capture_output=True, text=True).stdout.split(':')[2]
    except Exception: return None

def hf_download(repo, revision, dest, patterns=None, dry=False):
    """Download repo files (all, or the allow_patterns) at a pinned revision into dest: huggingface_hub API first, the hf /
    huggingface-cli binaries as a fallback, otherwise one clear line instead of a traceback."""
    what = f'{repo}@{revision[:12]}' + (f' ({len(patterns)} files)' if patterns else ' (all files)')
    if dry: print(f'download (dry run): {what} -> {dest}'); return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        exe = shutil.which('hf') or shutil.which('huggingface-cli')
        if not exe: sys.exit('missing dependency: the huggingface_hub Python package (pip install huggingface_hub) or the hf command on PATH')
        cmd = [exe, 'download', repo, '--revision', revision, '--local-dir', str(dest)] + sum((['--include', f] for f in (patterns or [])), [])
        print('download:', ' '.join(cmd)); subprocess.run(cmd, check=True); return
    print(f'download: {what} -> {dest}')
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(dest), allow_patterns=patterns)

def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''): h.update(chunk)
    return h.hexdigest()

def verify_sums(d):
    bad, n = [], 0
    for line in (d / 'SHA256SUMS').read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2: continue
        want, rel = parts[0], parts[1].strip().lstrip('*'); n += 1
        if not (d / rel).exists() or sha256_file(d / rel) != want: bad.append(rel)
    if bad: sys.exit(f'weights: {len(bad)} of {n} files fail SHA256SUMS in {d}: {bad[:5]}')
    print(f'weights: SHA256SUMS verified ({n} files)')

def ensure_weights(d, dry, repo=None, revision=None):
    d = pathlib.Path(d)
    if (d / 'SHA256SUMS').exists() and (d / 'model.safetensors').exists():
        print(f'weights: {d} (present; delete SHA256SUMS to force a re-download, or re-verify with `sha256sum -c SHA256SUMS` there)'); return d
    hf_download(repo or COMMON['hf_repo'], revision or COMMON['hf_revision'], d, dry=dry)
    if not dry: verify_sums(d)
    return d


def ensure_base_cfg(d, dry):
    """Base model's config + tokenizer files (pinned revision) next to the weights; the container export does not carry them."""
    d = pathlib.Path(d)
    if all((d / f).exists() for f in COMMON['base_cfg_files']): return d
    hf_download(COMMON['base_model'], COMMON['base_model_revision'], d, patterns=list(COMMON['base_cfg_files']), dry=dry)
    return d

def gtt_limits_gib(rocr_ids):
    """(limit, free) GiB of GTT for the host pools: per chosen AMD GPU (vendor 0x1002, matched by unique id) mem_info_gtt_total, capped
    by TTM's global page limit (ttm.pages_limit when set, else the kernel default of half of MemTotal); free = limit minus the GTT that
    is already in use on that GPU (other processes). The smallest over the GPUs; (None, None) when unreadable."""
    want = {g.lower().replace('gpu-', '') for g in rocr_ids}; devs = []
    for dev in pathlib.Path('/sys/class/drm').glob('card[0-9]*/device'):
        try:
            if (dev / 'vendor').read_text().strip() != '0x1002': continue
            if (dev / 'unique_id').read_text().strip().lower().replace('0x', '') in want:
                devs.append((int((dev / 'mem_info_gtt_total').read_text()), int((dev / 'mem_info_gtt_used').read_text())))
        except (OSError, ValueError): continue
    if not devs: return None, None
    try: ttm = int(pathlib.Path('/sys/module/ttm/parameters/pages_limit').read_text()) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError): ttm = 0
    if ttm <= 0:
        try: ttm = int(re.search(r'MemTotal:\s+(\d+)', open('/proc/meminfo').read()).group(1)) * 1024 // 2
        except Exception: ttm = 0
    lim = [min(t, ttm) if ttm > 0 else t for t, _ in devs]
    return min(lim) / (1 << 30), min(l - u for l, (_, u) in zip(lim, devs)) / (1 << 30)

def host_pool_env(choice, gpus, tier_gib=0.0, pinned_gib=None, ram_tier_gib=0.0, ram_tier_overhead_gib=0.0):
    """Host pools of a start; returns (env, RAM-tier GiB that fits). All pinned host memory of both TP workers is charged to one GPU's GTT
    when HSA_USERPTR_FOR_PAGED_MEM=0. Modes with a measured pinned_gib (4-bit: host tables + expert host tier, GTT at ready) use the ladder:
      GTT free >= pinned + RAM tier + 2 GiB -> GTT pools + the RAM tier;  >= pinned + 2 GiB -> GTT pools, RAM tier shrunk to what fits
      (>= 4 GiB) or off;  less -> ROCm user-pointer pools, no RAM tier, with a warning.
    Other modes: GTT pools when the host tables (+ margin) fit, else user-pointer pools (the published 3-bit behaviour)."""
    hp = COMMON['host_pools']; lim, free = gtt_limits_gib(gpus)
    margin = 2.0 if pinned_gib else hp['gtt_margin_gib']; base = (pinned_gib or hp['gtt_need_gib'] + tier_gib) + margin
    fits = free is not None and free >= base
    if choice == 'gtt' and not fits:
        sys.exit(f'host pools: --host-pools gtt needs {base:.1f} GiB of free GTT on the first GPU, this host has '
                 f'{"an unknown amount" if free is None else f"{free:.1f} GiB (limit {lim:.1f} GiB)"} (use --host-pools userptr)')
    if choice == 'auto': choice = 'gtt' if fits else 'userptr'
    rt = 0.0
    if choice == 'gtt' and ram_tier_gib:
        head = free - base - ram_tier_overhead_gib   # the tier's GTT at load: its size plus the measured overhead (cache mode: ~106 GiB at 8 GiB)
        rt = ram_tier_gib if head >= ram_tier_gib else (head if head >= 4.0 else 0.0)
        if rt < ram_tier_gib: print(f'RAM tier {"shrunk to " + f"{rt:.1f} GiB" if rt else "disabled"}: GTT free {free:.1f} GiB < needed {base + ram_tier_overhead_gib + ram_tier_gib:.1f} GiB')
    print(f'host pools: {choice} (GTT limit {"unknown" if lim is None else f"{lim:.1f} GiB"}, free {"unknown" if free is None else f"{free:.1f} GiB"}, '
          f'needed {base:.1f} GiB for GTT pools)' + ('' if choice == 'gtt' else '; user-pointer pools can stall when the kernel compacts host memory'))
    if choice == 'userptr' and pinned_gib:
        busy = '' if free is None or lim is None or lim - free < 1 else (f' ({lim - free:.0f} GiB of the {lim:.0f} GiB GTT limit is in use by other '
                                                                         'processes: stop other GPU servers first if that is not intended)')
        print(f'WARNING host pools: host GTT {"unknown" if free is None else f"{free:.0f} GiB"} < {base:.0f} GiB{busy}: this mode was benchmarked with its '
              f'~{pinned_gib:.0f} GiB of pinned host memory in GTT (by default the GTT limit is half the RAM, so in practice a '
              f'{128 if pinned_gib < 60 else 256} GB host). Using '
              'pageable pinned (user-pointer) memory instead, without a RAM tier: it starts and runs, but slower than the benchmark (186 vs 192 tok/s, '
              '389 vs 423 tok/s at 8 requests in our validation) and expect GPU stalls of up to several seconds when the kernel compacts host memory. '
              'Raising the GTT limit (amdgpu.gttsize / ttm.pages_limit) is a system setting you may choose; this launcher never changes it.')
    return (dict(hp['env']) if choice == 'gtt' else {}), rt

def pgmigrate_success():
    try: return int(re.search(r'^pgmigrate_success (\d+)$', open('/proc/vmstat').read(), re.M).group(1))
    except Exception: return None

def drop_page_cache(root):
    """posix_fadvise(DONTNEED) on every checkpoint file: drops their clean page-cache pages after the load (frees host memory, so the
    kernel compacts less while the server is serving). Read-only, nothing is written; files in use elsewhere are simply re-read."""
    n = b = 0
    for p in pathlib.Path(root).rglob('*'):
        try:
            if not p.is_file() or p.is_symlink(): continue
            fd = os.open(p, os.O_RDONLY)
            try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1; b += os.fstat(fd).st_size
            finally: os.close(fd)
        except (OSError, AttributeError): continue
    return n, b

def wait_ready(port, proc, name, timeout_s=1800, settle_s=30.0, quiet_s=5.0, rate_max=10000, weights=None, drop_cache=True,
               host='0.0.0.0'):
    """Ready = /health answers, then (after dropping the checkpoint's page cache) the host's page migration rate (/proc/vmstat
    pgmigrate_success) stays below rate_max pages/s for quiet_s seconds, capped at settle_s: the kernel compacts memory right after
    the engine is ready, which can pause the GPU queues for seconds; the first requests then land after it. Returns True when ready."""
    import time, urllib.request
    wildcard = host in ('0.0.0.0', '::', '')            # every interface, loopback included
    probe = '127.0.0.1' if wildcard else (f'[{host}]' if ':' in host else host)
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(f'http://{probe}:{port}/health', timeout=3) as r:
                if r.status == 200: break
        except Exception: pass
        if proc is not None and proc.poll() is not None: return False
        if proc is None and subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', name], capture_output=True, text=True).stdout.strip() != 'true': return False
        if time.time() - t0 > timeout_s: print(f'launcher: /health did not answer within {timeout_s} s', flush=True); return False
        time.sleep(2)
    t_health = time.time()
    if drop_cache and weights is not None:
        n, b = drop_page_cache(weights); print(f'launcher: dropped the page cache of {n} checkpoint files ({b / 2**30:.1f} GiB)', flush=True)
    if settle_s > 0 and pgmigrate_success() is not None:
        t1 = time.time(); last = pgmigrate_success(); tl = t1; calm = 0.0; peak = 0.0
        while time.time() - t1 < settle_s:
            time.sleep(0.5); now = time.time(); cur = pgmigrate_success(); rate = (cur - last) / max(now - tl, 1e-3); last, tl = cur, now
            peak = max(peak, rate); calm = calm + 0.5 if rate < rate_max else 0.0
            if calm >= quiet_s: break
        print(f'launcher: settle wait {time.time() - t1:.1f} s (host page migration peak {peak:,.0f} pages/s; quiet = below {rate_max:,} pages/s '
              f'for {quiet_s:g} s, cap {settle_s:g} s)', flush=True)
    print(f'launcher: READY on port {port} ({time.time() - t0:.0f} s after start, {time.time() - t_health:.0f} s after /health)', flush=True)
    print(f'launcher: API http://{probe}:{port}/v1' + (f' (or http://<this-machine-ip>:{port}/v1 from another system)' if wildcard else ''),
          flush=True)
    print('launcher: every 10 s the server log shows "Avg prompt throughput" (reading prompts, prefill) and "Avg generation throughput" '
          '(writing tokens); a long prompt shows up as prompt throughput, not as slow generation (README: Troubleshooting)', flush=True)
    return True

def tier_params(spec):
    """'static:vram_gib=21.6,counts=...,pool=1,core=0.3' -> (kind, {key: value})"""
    kind, _, rest = spec.partition(':')
    return kind, dict(kv.split('=', 1) for kv in rest.split(',') if '=' in kv)

def tier_spec(spec, vram_gib=None, host_pool=None):
    kind, kv = tier_params(spec)
    if vram_gib is not None: kv['vram_gib'] = f'{vram_gib:g}'
    if host_pool is not None: kv['pool'] = '1' if host_pool == 'on' else '0'
    return kind + ':' + ','.join(f'{k}={v}' for k, v in kv.items())

def pinned_host_gib(spec, expert_gib_total, ranks):
    """Pinned host memory the expert tier allocates (both ranks): pool mode keeps a host copy of every non-core expert;
    without the pool only the experts beyond the VRAM budget live on the host."""
    _, kv = tier_params(spec); per_rank = expert_gib_total / ranks
    if kv.get('pool', '0') == '1': host = per_rank * (1 - float(kv.get('core', '0')))
    else: host = max(0.0, per_rank - float(kv.get('vram_gib', per_rank)))
    return host * ranks

def meminfo_gib(field):
    """/proc/meminfo field (MemTotal, MemAvailable) in GiB; None when unreadable."""
    try: return int(re.search(rf'{field}:\s+(\d+)', open('/proc/meminfo').read()).group(1)) / (1 << 20)
    except Exception: return None

def ram_need_gib(tier_gib, pinned_gib=None, pageable_gib=0.0):
    """Host memory a start keeps resident: pinned (host tables + expert host tier, or the measured pinned total) + pageable PLE tables."""
    tables = COMMON['host_pools']['gtt_need_gib'] - pageable_gib
    return max(tables + tier_gib, pinned_gib or 0.0) + pageable_gib

def ram_check(tier_gib, margin_gib=8.0, pinned_gib=None, pageable_gib=0.0):
    """Pinned host memory of a start: the host tables (every mode) plus the expert host tier (4-bit modes), against MemAvailable;
    with --ple-host pageable the PLE tables (part of the host tables) move from pinned to pageable host memory and still need the RAM."""
    tables = COMMON['host_pools']['gtt_need_gib'] - pageable_gib; need_gib = ram_need_gib(tier_gib, pinned_gib, pageable_gib)
    avail = meminfo_gib('MemAvailable')
    if avail is None: print('ram: /proc/meminfo unreadable; not checked'); return []
    ok = avail >= need_gib + margin_gib
    print(f'ram: pinned host memory {need_gib - pageable_gib:.1f} GiB (host tables {tables:g}' + (f' + expert host tier {tier_gib:.1f}' if tier_gib else '') + ', both GPUs)'
          + (f' + pageable PLE tables {pageable_gib:.1f} GiB' if pageable_gib else '') + '; '
          f'available {avail:.1f} GiB; needed {need_gib + margin_gib:.1f} GiB with a {margin_gib:g} GiB margin -> {"ok" if ok else "NOT ENOUGH"}')
    return [] if ok else [f'{need_gib + margin_gib:.0f} GiB of available RAM for the pinned host memory (have {avail:.0f} GiB'
                          + ')']

def p2p_check(rocr_ids):
    """Warn (never refuse) when the chosen GPUs have no peer-to-peer link in the KFD topology: tensor-parallel traffic then goes
    through host memory and decode/prefill run slower than measured. Uses /sys/class/kfd topology (gpu_id per node, p2p_links) and,
    when readable, the kernel's CONFIG_HSA_AMD_P2P; nothing host-specific is assumed."""
    import gzip
    topo = pathlib.Path('/sys/class/kfd/kfd/topology/nodes')
    try:
        gpu_nodes = {n.name for n in topo.iterdir() if (n / 'gpu_id').read_text().strip() not in ('', '0')}
    except OSError:
        print('p2p: KFD topology unreadable; not checked'); return
    linked = set()
    for n in gpu_nodes:
        for l in (topo / n / 'p2p_links').glob('*/properties'):
            try:
                kv = dict(line.split() for line in l.read_text().splitlines() if len(line.split()) == 2)
                if kv.get('node_to') in gpu_nodes: linked.add(n)
            except OSError: continue
    cfg = None
    for src in (f'/boot/config-{os.uname().release}', '/proc/config.gz'):
        try:
            txt = gzip.open(src, 'rt').read() if src.endswith('.gz') else open(src).read()
            cfg = 'y' if 'CONFIG_HSA_AMD_P2P=y' in txt else 'not set'; break
        except OSError: continue
    if len(gpu_nodes) >= 2 and linked >= gpu_nodes:
        print(f'p2p: GPU peer-to-peer links present ({len(gpu_nodes)} GPU nodes)' + (f'; kernel CONFIG_HSA_AMD_P2P={cfg}' if cfg else ''))
    else:
        print(f'p2p: WARNING: no GPU peer-to-peer link between the GPUs in the KFD topology' + (f' (kernel CONFIG_HSA_AMD_P2P={cfg})' if cfg else '')
              + '; tensor-parallel traffic goes through host memory and both GPUs run slower than our numbers. Check the kernel option, '
              'the BIOS (Above 4G decoding, Resizable BAR) and that both GPUs share a PCIe root without ACS blocking peer access.')

def prerequisites(tp):
    """What a user needs; prints exactly what is missing (also in --dry-run) and stops unless everything is there."""
    missing = []
    if sys.version_info < (3, 10): missing.append(f'python3 >= 3.10 (this is {sys.version.split()[0]})')
    if not shutil.which('docker'): missing.append('docker (the docker command is not on PATH)')
    else:
        r = subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], capture_output=True, text=True)
        if r.returncode != 0: missing.append('docker daemon access (`docker info` failed: ' + (r.stderr.strip().splitlines() or ['no details'])[-1][:100] + ')')
    try: import huggingface_hub  # noqa: F401
    except ImportError:
        if not (shutil.which('hf') or shutil.which('huggingface-cli')): missing.append('the huggingface_hub Python package: pip install huggingface_hub')
    if not os.path.exists('/dev/kfd'): missing.append('/dev/kfd (ROCm kernel driver not loaded)')
    gpus = amd_gpus()
    if len(gpus) < tp: missing.append(f'{tp} AMD GPUs visible (found {len(gpus)}: {gpus})')
    print('prerequisites:', 'all present' if not missing else 'MISSING -> ' + '; '.join(missing))
    return missing

def ensure_view(weights, view, base_cfg, dry, wanted_optional=(), int6=False):
    """Runtime view for vLLM (as the benchmarks used): base config with the Paiton quantization block + tokenizer files, and links
    to the container's safetensors as the container will see them (/models/ck/...). Written next to the weights, never inside them."""
    weights, view = pathlib.Path(weights), pathlib.Path(view); ck = COMMON['mounts']['checkpoint']; vdir = COMMON['mounts']['view']
    layers = sorted(p.name for p in (weights / 'layers').glob('L[0-9][0-9].safetensors'))
    links = {n: f'{ck}/layers/{n}' for n in layers}
    links.update({'model.safetensors': f'{ck}/model.safetensors', 'manifest.json': f'{ck}/manifest.json', 'ple-e4m3': f'{ck}/common/ple-e4m3',
                  'mtp-vision.safetensors': f'{ck}/common/bf16-mtp-vision.safetensors', 'mtp-experts.safetensors': f'{ck}/mtp/mtp-experts.safetensors',
                  'mtp-draft-int2/draft-head-int2.safetensors': f'{ck}/mtp/draft-head-int2.safetensors', 'mtp-draft-int2/draft-head-int2.json': f'{ck}/mtp/draft-head-int2.json'})
    # INT6 trunk overlay (top-level i6-LNN files): only for modes that read it (PAITON_FN_INT6_EXTRA). Other modes never see it: a loader
    # without INT6 support fails on those tensors (11 Oct: the published 3-bit image on a weights dir that also holds trunk-int6/)
    for f in (sorted((weights / 'trunk-int6' / 'layers').glob('L[0-9][0-9].safetensors')) if int6 else ()):
        links['i6-' + f.name] = f'{ck}/trunk-int6/layers/{f.name}'
    for k, rel in COMMON.get('view_links_if_present', {}).items():   # files of some exports only (the hybrid 4-bit format map)
        if (weights / rel).exists(): links[k] = f'{ck}/{rel}'
    for k, rel in COMMON.get('view_links_optional', {}).items():   # files only the modes that list them in 'requires' load
        if rel in wanted_optional and (weights / rel).exists(): links[k] = f'{ck}/{rel}'
    missing = [tgt for tgt in links.values() if not (weights / tgt[len(ck) + 1:]).exists()]
    if missing and dry: print(f'view (dry run): built after the download ({len(missing)} files not present yet)'); return view
    if missing: sys.exit(f'weights incomplete, missing: {missing[:4]}')
    have = {str(x.relative_to(view)) for x in view.rglob('*') if x.is_symlink()} if view.exists() else set()
    if (view / 'config.json').exists() and have == set(links): print(f'view: {view} (present)'); return view
    print(f'view: building {view} ({len(layers)} layers)')
    if dry: return view
    if view.exists(): shutil.rmtree(view)
    view.mkdir(parents=True); (view / 'mtp-draft-int2').mkdir()
    cfg = json.loads((base_cfg / 'config.json').read_text()); cfg['language_model_only'] = True
    cfg['quantization_config'] = {'quant_method': 'paiton_fnq', 'view_dir': vdir}
    (view / 'config.json').write_text(json.dumps(cfg, indent=2))
    for f in COMMON['base_cfg_files']:
        if f != 'config.json' and (base_cfg / f).exists(): shutil.copy(base_cfg / f, view / f)
    for k, tgt in links.items(): os.symlink(tgt, view / k)
    return view

def set_alloc_fraction(env, frac):
    """PyTorch's per-process VRAM cap in PYTORCH_ALLOC_CONF (per_process_memory_fraction), other allocator options kept."""
    parts = [x for x in env.get('PYTORCH_ALLOC_CONF', '').split(',') if x and not x.startswith('per_process_memory_fraction:')]
    env['PYTORCH_ALLOC_CONF'] = ','.join(parts + [f'per_process_memory_fraction:{frac:g}'])

def kv_pool_check(mode, m, engine_args):
    """Warn when the KV cache is small for how the mode is used (GitHub issue #22): with prefix caching, a cache smaller than twice
    the window means agent requests queue and an evicted conversation is re-processed from the start; a lowered
    --kv-cache-memory-bytes shrinks the cache (and below one window vLLM refuses to start)."""
    g = lambda f: int(engine_args[engine_args.index(f) + 1]) if f in engine_args else None  # noqa: E731
    window, kv_bytes = g('--max-model-len'), g('--kv-cache-memory-bytes')
    mode_bytes = int(m['engine_args'][m['engine_args'].index('--kv-cache-memory-bytes') + 1])
    base = m.get('kv_pool_tokens')
    if not (window and kv_bytes and base):
        return None
    tokens = int(base * kv_bytes / mode_bytes)
    if kv_bytes < mode_bytes:
        print(f'WARNING KV cache: --kv-cache-memory-bytes {kv_bytes:,} is below the {mode} value {mode_bytes:,}: the cache holds about '
              f'{tokens:,} tokens instead of {base:,}' + (f'; that is less than one {window:,}-token window, so the server will refuse to start'
                                                            if tokens < window else ''))
    if '--enable-prefix-caching' in engine_args and tokens < 2 * window:
        print(f'WARNING prefix caching: the KV cache holds about {tokens // 1000}K tokens < 2 x the {window // 1000}K window: agent requests '
              'will queue, and a conversation evicted from the cache re-processes the whole conversation on its next turn'
              + (' (--mode 4bit-cache keeps evicted prefixes in a RAM tier)' if mode.startswith('decode') or mode.startswith('prefill') else ''))
    return tokens


def low_ram_plan(a, m, image, gpus):
    """--ple-host auto on the 4-bit modes: PLE tables pinned (as benchmarked) when the GPU's GTT limit (half of MemTotal by default)
    holds the measured pinned total + 2 GiB; else pageable, on modes where it is measured and with an image that has the pageable
    runtime. An explicit flag is kept; one line is printed when the launcher picks pageable."""
    if not m.get('expert_tier'):
        if a.ple_host == 'auto': a.ple_host = 'pinned'
        return
    ph = COMMON['ple_host']
    if a.ple_host == 'auto':
        a.ple_host = 'pinned'
        lim, _ = gtt_limits_gib(gpus); need = m['gtt_pinned_gib'] + 2.0
        if lim is not None and lim < need:
            if 'gtt_pinned_gib_pageable' not in m:
                print(f'low-RAM host: GTT limit {lim:.0f} GiB < {need:.0f} GiB for the pinned PLE tables; --ple-host pageable is not measured on '
                      f'--mode {a.mode} yet (use --mode 4bit or --mode 4bit-cache on this host); keeping them pinned')
            elif not any(ref in image for ref in ph['images']):
                print(f'low-RAM host: GTT limit {lim:.0f} GiB < {need:.0f} GiB for the pinned PLE tables, but {image} has no pageable PLE runtime; '
                      'keeping them pinned')
            else:
                a.ple_host = 'pageable'
                print(f'low-RAM host: GTT limit {lim:.0f} GiB < {need:.0f} GiB for the pinned PLE tables (MemTotal '
                      f'{meminfo_gib("MemTotal") or 0:.0f} GiB) -> --ple-host pageable automatically (pinned host memory ~{m["gtt_pinned_gib_pageable"]:.0f} '
                      'GiB, same output; --ple-host pinned keeps the benchmarked layout)')

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='decode', choices=sorted(MODES)); ap.add_argument('--port', type=int, default=18982); ap.add_argument('--name', default='paiton-flashnext')
    ap.add_argument('--host', default='0.0.0.0', help='address the API listens on (default 0.0.0.0, every interface; 127.0.0.1 keeps it on this machine)')
    ap.add_argument('--api-key', default=None, metavar='KEY', help='opt-in: the server then requires this key on /v1 requests (handed to Docker as VLLM_API_KEY by '
                    'variable name, never printed). Preferred: set PAITON_API_KEY in the environment instead (the default here); a key '
                    'given with --api-key is visible in the process list (ps). Default: no key')
    ap.add_argument('--weights', default=None, help='weights dir (default per mode: ~/paiton-models/Qwen3.8-Flash-Next-W3A8, 4-bit modes ~/paiton-models/Qwen3.8-Flash-Next-W4A8)');
    ap.add_argument('--kv-cache-memory-bytes', type=int, default=None, help='KV cache size per GPU in bytes (default: the mode value); '
                    'lower values shrink the cache the prefix cache and long conversations live in')
    ap.add_argument('--expert-vram-gib', type=float, default=None, help='4-bit modes: VRAM per GPU for resident experts (default from modes.json)'); ap.add_argument('--host-pool', choices=('on', 'off'), default=None, help='4-bit modes: keep a pinned host copy of every non-core expert (on, default) or only of the non-resident ones (off)'); ap.add_argument('--cache', default=None, help='compile cache dir (default ~/.cache/paiton/flashnext/<mode>)')
    ap.add_argument('--image', default=None, help='image reference (default: the pinned release image)'); ap.add_argument('--devices', default=None, help='comma list of ROCR ids (default: all AMD GPUs; two are required)')
    ap.add_argument('--view', default=None, help='runtime view dir (default: <weights>-view)'); ap.add_argument('--base-cfg', default=None, help='dir with the base model config + tokenizer files (default: <weights>-base-cfg, downloaded at the pinned revision)')
    ap.add_argument('--ple-host', choices=('auto', 'pinned', 'pageable'), default='auto', help='4-bit modes 4bit and 4bit-cache: where the per-layer '
                    'embedding tables live on the host: pinned (as benchmarked) or pageable (~49 GiB less pinned host memory and GTT, so the GTT '
                    'pools fit a 128 GB host; same output, decode within ~1 %%; needs an image with the pageable PLE runtime); auto (default): '
                    'pageable when the GTT limit cannot hold the pinned tables, else pinned')
    ap.add_argument('--prefix-caching', action='store_true', help='opt-in: align-mode prefix caching (recurrent state cached per 2048-token block); see RELEASE-NOTES.md'); ap.add_argument('--served-model-name', default='Qwen3.8-Flash-Next'); ap.add_argument('--host-pools', choices=('auto', 'gtt', 'userptr'), default='auto', help='host-resident tables: driver-owned GTT memory or ROCm user-pointer memory; auto = the mode default (3-bit: userptr; 4-bit: gtt when the GTT size holds them)'); ap.add_argument('--detach', action='store_true'); ap.add_argument('--settle-s', type=float, default=30.0, help='after /health: wait (at most this long) for the host page migration to calm down before reporting READY; 0 = off'); ap.add_argument('--keep-page-cache', action='store_true', help='do not drop the checkpoint files from the page cache after the load'); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(); m = MODES[a.mode]
    if a.api_key:
        print('note: --api-key is visible in the process list (ps) of this machine; prefer PAITON_API_KEY=<key> in the environment')
    a.api_key = a.api_key or os.environ.get('PAITON_API_KEY') or None
    if a.api_key: print('API key: required on /v1 requests (value not printed)')
    wspec = COMMON.get(m['weights_key']) if m.get('weights_key') else None
    if a.weights is None: a.weights = os.path.expanduser((wspec or {}).get('default_dir', '~/paiton-models/Qwen3.8-Flash-Next-W3A8'))
    if m.get('status', '').startswith('PENDING') : print(f'mode {a.mode}: {m["status"]}')
    ispec = COMMON[m.get('image_key', 'image')]
    image = a.image or (ispec['tag'] + ('@' + ispec['digest'] if ispec['digest'] != 'PIN-ME' else ''))
    gpus = a.devices.split(',') if a.devices else amd_gpus()
    low_ram_plan(a, m, image, gpus[:COMMON['tp']])
    pinned_gib = m.get('gtt_pinned_gib'); pageable_gib = 0.0
    if a.ple_host == 'pageable':
        if 'gtt_pinned_gib_pageable' not in m:
            sys.exit(f'--ple-host pageable is not measured on --mode {a.mode}: use --mode '
                     + ' or --mode '.join(k for k, v in MODES.items() if 'gtt_pinned_gib_pageable' in v))
        pinned_gib = m['gtt_pinned_gib_pageable']; pageable_gib = COMMON['ple_host']['pageable_gib_both_ranks']
    ph = COMMON['ple_host']
    if a.ple_host == 'pageable' and not any(ref in image for ref in ph['images']):
        msg = (f'--ple-host pageable needs an image with the pageable PLE runtime ({", ".join(ph["images"]) or "none published yet"}); '
               f'{image} keeps the PLE tables pinned (about 96 GiB of pinned host memory)')
        if a.dry_run: print('WARNING:', msg)
        else: sys.exit(msg)
    missing = prerequisites(COMMON['tp'])
    tier = None; tier_gib = 0.0
    if m.get('expert_tier'):
        et = m['expert_tier']
        if a.host_pool == 'off' and tier_params(m['env'].get(et['env'], et['default']))[1].get('classes') == '1':
            sys.exit(f'--host-pool off does not work with --mode {a.mode}: its expert tier swaps experts through the host pool (classes=1 '
                     'needs pool=1); keep the default')
        tier = tier_spec(m['env'].get(et['env'], et['default']), a.expert_vram_gib, a.host_pool)
        tier_gib = pinned_host_gib(tier, et['expert_bytes_gib_both_ranks'], COMMON['tp'])
        dflt_gib = pinned_host_gib(tier_spec(m['env'].get(et['env'], et['default'])), et['expert_bytes_gib_both_ranks'], COMMON['tp'])
        if pinned_gib and abs(tier_gib - dflt_gib) > 0.05:   # the measured pinned totals hold the default expert tier
            pinned_gib = round(pinned_gib - dflt_gib + tier_gib, 1)
    missing += ram_check(tier_gib, 16.0 if pinned_gib else 8.0, pinned_gib, pageable_gib)
    if m.get('expert_tier'):
        tot = meminfo_gib('MemTotal')
        if pageable_gib:
            print(f'host RAM: {"unknown" if tot is None else f"{tot:.0f} GiB"} total; --ple-host pageable keeps about {pinned_gib:.0f} GiB of pinned host '
                  f'memory in one GPU\'s GTT (about {pinned_gib + 2:.0f} GiB of free GTT; the GTT limit is half the RAM by default: a 128 GB host) and '
                  f'the {pageable_gib:.0f} GiB of PLE tables in pageable memory; see the host pools line below')
        else:
            print(f'host RAM: {"unknown" if tot is None else f"{tot:.0f} GiB"} total; the benchmarked 4-bit configuration keeps about 96 GiB of pinned '
                  'host memory in one GPU\'s GTT, so it needs about 98 GiB of free GTT (the GTT limit is half the RAM by default: a 256 GB host); '
                  'see the host pools line below' + ('; --ple-host pageable needs about 49 GiB less' if 'gtt_pinned_gib_pageable' in m else ''))
    if missing and not a.dry_run: sys.exit('fix the missing prerequisites above and run again')
    for rel in m.get('requires', []):
        if not (pathlib.Path(a.weights) / rel).exists() and not a.dry_run: sys.exit(f'mode {a.mode} needs {rel} in the checkpoint dir (not in this copy of the repo)')
    if len(gpus) < COMMON['tp'] and not a.dry_run: sys.exit(f'need {COMMON["tp"]} AMD GPUs, found {gpus}')
    gpus = gpus[:COMMON['tp']]
    p2p_check(gpus)
    weights = ensure_weights(a.weights, a.dry_run, *((wspec['hf_repo'], wspec['hf_revision']) if wspec else (None, None)))
    want_rel = (wspec or COMMON)['hf_repo'].split('/')[-1]
    try: got_rel = json.loads((weights / 'config.json').read_text()).get('paiton_release')
    except Exception: got_rel = None
    if got_rel and got_rel != want_rel:
        msg = f'mode {a.mode} expects the {want_rel} weights, but {weights} holds {got_rel}; pass --weights with the matching download'
        if a.dry_run: print('WARNING:', msg)
        else: sys.exit(msg)
    base_cfg = ensure_base_cfg(a.base_cfg or str(weights) + '-base-cfg', a.dry_run); view = ensure_view(weights, a.view or str(weights) + '-view', base_cfg, a.dry_run, tuple(m.get('requires', [])), int6=bool(m['env'].get('PAITON_FN_INT6_EXTRA')))
    cache = pathlib.Path(a.cache or os.path.expanduser(f'~/.cache/paiton/flashnext/{a.mode}')); cache.mkdir(parents=True, exist_ok=True)
    env = dict(m['env']); env['VLLM_CACHE_ROOT'] = COMMON['mounts']['cache']; env['HF_HUB_OFFLINE'] = '1'
    if tier: env[m['expert_tier']['env']] = tier
    engine_args = list(m['engine_args'])
    if a.prefix_caching and '--enable-prefix-caching' not in engine_args and m.get('expert_tier'):
        sys.exit(f'--prefix-caching is not measured on --mode {a.mode}: use --mode 4bit-cache (prefix caching + RAM prefix tier)')
    if a.prefix_caching and '--enable-prefix-caching' in engine_args: print(f'mode {a.mode}: prefix caching is part of this mode (--prefix-caching not needed)')
    elif a.prefix_caching:
        pc = COMMON['prefix_caching']; engine_args = [x for x in engine_args if x not in pc['engine_args_drop']] + pc['engine_args_add']; env.update(pc['env'])
        if m.get('alloc_fraction_prefix_caching'):   # prefix caching has its own measured allocator cap (per mode)
            set_alloc_fraction(env, m['alloc_fraction_prefix_caching'])
    env['ROCR_VISIBLE_DEVICES'] = ','.join(gpus); env['HIP_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(len(gpus))); env['CUDA_VISIBLE_DEVICES'] = env['HIP_VISIBLE_DEVICES']
    env['HOME'] = '/tmp'; env['USER'] = env['LOGNAME'] = 'paiton'
    for k in COMMON['host_pools']['env']: env.pop(k, None)   # a mode profile may carry it; the host-pools choice decides
    if a.kv_cache_memory_bytes is not None:
        i = engine_args.index('--kv-cache-memory-bytes'); engine_args[i + 1] = str(a.kv_cache_memory_bytes)
    kv_pool_check(a.mode, m, engine_args)
    rtier = m.get('ram_tier') or {}
    hp_choice = m.get('host_pools_default', 'auto') if a.host_pools == 'auto' else a.host_pools
    if a.host_pools == 'auto' and hp_choice == 'userptr':
        print('host pools: userptr (the default of this mode, as published); --host-pools gtt removes the occasional memory pauses under load '
              'and costs ~1 GiB VRAM in decode')
    hp_env, rt = host_pool_env(hp_choice, gpus, tier_gib, pinned_gib=pinned_gib, ram_tier_gib=float(rtier.get('gib', 0.0)),
                               ram_tier_overhead_gib=float(rtier.get('gtt_overhead_gib', 0.0)))
    env.update(hp_env)
    for k in ph['env']: env.pop(k, None)
    if a.ple_host == 'pageable':
        env.update(ph['env']); print(f'PLE tables: pageable host memory ({pageable_gib:.1f} GiB, both GPUs; not pinned, not in GTT)')
    if rtier:   # RAM prefix tier (cache mode): prefix-cache blocks evicted from VRAM are kept in pinned host memory (GTT) and loaded back on a hit
        rt = int(rt)
        if rt >= rtier.get('min_gib', 4.0):
            kvt = dict(rtier['kv_transfer_config']); kvt['kv_connector_extra_config'] = dict(kvt['kv_connector_extra_config'], cpu_bytes_to_use=rt << 30)
            engine_args += ['--kv-transfer-config', json.dumps(kvt)]
            print(f'RAM prefix tier: {rt} GiB of pinned host memory')
        else:
            for k in rtier['env']: env.pop(k, None)
            print(f'WARNING RAM prefix tier: off (it needs {rtier["gib"] + rtier.get("gtt_overhead_gib", 0.0):g} GiB of free GTT beyond the host pools); '
                  'prefix caching stays on, in VRAM only')
    env['PAITON_FN_PLACEMENT'] = f'counts:{COMMON["mounts"]["checkpoint"]}/common/routing-counts.json'   # the container runs as the host uid, which has no passwd entry: getpass.getuser() needs these
    cmd = ['docker', 'run', '--rm', '--name', a.name, '--user', f'{os.getuid()}:{os.getgid()}'] + COMMON['docker'] + (ph['docker_add'] if a.ple_host == 'pageable' else [])
    gid = render_gid(); cmd += ['--group-add', gid] if gid else []
    if a.detach: cmd += ['-d']
    for k, v in sorted(env.items()): cmd += ['-e', f'{k}={v}']
    if a.api_key: cmd += ['-e', 'VLLM_API_KEY']   # by name: Docker reads the value from the launcher's environment (below)
    cmd += ['-v', f'{weights}:{COMMON["mounts"]["checkpoint"]}:ro', '-v', f'{view}:{COMMON["mounts"]["view"]}:ro', '-v', f'{cache}:{COMMON["mounts"]["cache"]}']
    if env.get('PAITON_FN_KV_TIER') == '1':
        # RAM prefix tier: its boot hook must sit in site-packages at interpreter start (every engine and worker process), or stock
        # vLLM offloading rejects the route-1 scratch ring group at init. The container runs as the host uid and cannot write there,
        # so the launcher writes the one-line hook next to the cache dir (not inside it: an empty cache gets the seed) and mounts it.
        hook = cache.with_name(cache.name + '-kv-tier.pth')
        if not a.dry_run: hook.write_text(KV_TIER_PTH + '\n')
        cmd += ['-v', f'{hook}:{SITE_PACKAGES}/zz_paiton_fn_kv_tier.pth:ro']
    # first start of a mode: copy the baked compile-cache seed into the (empty) cache, then serve
    inner = (f'if [ -d /opt/paiton/cache-seed/{m.get("seed", a.mode)} ] && [ -z "$(ls -A {COMMON["mounts"]["cache"]} 2>/dev/null)" ]; then cp -a /opt/paiton/cache-seed/{m.get("seed", a.mode)}/. {COMMON["mounts"]["cache"]}/; fi; '
             f'exec python3 -m vllm.entrypoints.openai.api_server --model {COMMON["mounts"]["view"]} --tokenizer {COMMON["mounts"]["view"]} '
             + ' '.join(shlex.quote(x) for x in engine_args) + f' --served-model-name {shlex.quote(a.served_model_name)} --host {shlex.quote(a.host)} --port {a.port}')
    cmd += ['--entrypoint', 'bash', image, '-c', inner]
    print(f'mode {a.mode}: image {image}; GPUs {gpus}; weights {weights}; cache {cache}; port {a.port}' + (f'; expert tier {tier}' if tier else ''))
    if a.dry_run: print(' '.join(shlex.quote(x) for x in cmd)); return
    ready = dict(port=a.port, name=a.name, settle_s=a.settle_s, weights=weights, drop_cache=not a.keep_page_cache, host=a.host)
    run_env = dict(os.environ, VLLM_API_KEY=a.api_key) if a.api_key else None
    if a.detach:
        rc = subprocess.run(cmd, env=run_env).returncode
        if rc: sys.exit(rc)
        sys.exit(0 if wait_ready(proc=None, **ready) else 1)
    proc = subprocess.Popen(cmd, env=run_env)
    import signal
    for sig in (signal.SIGTERM, signal.SIGHUP): signal.signal(sig, lambda n, f: proc.send_signal(n))   # the docker client stops the container
    try:
        wait_ready(proc=proc, **ready); sys.exit(proc.wait())
    except KeyboardInterrupt:
        try: sys.exit(proc.wait())
        except KeyboardInterrupt: proc.terminate(); sys.exit(proc.wait())

if __name__ == '__main__':
    main()
