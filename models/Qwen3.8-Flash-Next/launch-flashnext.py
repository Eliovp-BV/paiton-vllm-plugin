#!/usr/bin/env python3
"""Qwen3.8 Flash Next on two AMD R9700 (ROCm 10, vLLM 0.29, tensor parallel 2) with the Paiton plugin image.

Modes (modes.json, from the measured runs on this image):
  decode             default: MTP depth 3, 98,304-token window, KV 1.6 GiB per card              (decode / concurrency numbers)
  prefill-long       200,000-token window, MTP off, 3 GiB KV per card                            (prefill numbers)
  decode-nopf / prefill-long-nopf   the same with exact-arithmetic prefill (about 7 % slower)
  --prefix-caching   opt-in on any mode: align-mode prefix caching (byte-identical hits)
Host pools: the host-resident tables (~55 GiB, both ranks) are allocated as driver-owned GTT memory (HSA_USERPTR_FOR_PAGED_MEM=0)
  when the GPUs' GTT size holds them, so host memory compaction cannot stall the GPU queues; --host-pools userptr keeps ROCm's default.
  4bit / 4bit-200k   4-bit experts (W4A8 repo) with the expert host tier: experts beyond --expert-vram-gib per GPU live in
                     pinned host memory (about 41 GiB with the default host pool; the launcher checks the RAM first)
Prerequisites: python3 >= 3.10, docker with GPU device access (/dev/kfd, /dev/dri), the huggingface_hub Python package
  (`pip install huggingface_hub`; the `hf` / `huggingface-cli` binaries are used only as a fallback), two AMD GPUs visible.
Usage: launch-flashnext.py [--mode ...] [--prefix-caching] [--port 18982] [--name paiton-flashnext] [--detach] [--dry-run]
       [--weights DIR] [--cache DIR] [--image REF] [--devices GPU-xxx,GPU-yyy] [--served-model-name Qwen3.8-Flash-Next]
Weights: downloaded once from the Hugging Face repo in modes.json (pinned revision, SHA256SUMS verified) into --weights.
"""
import argparse, json, os, pathlib, re, shlex, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
CFG = json.loads((HERE / 'modes.json').read_text()); COMMON = CFG['common']; MODES = CFG['modes']

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

def host_pool_env(choice, gpus, tier_gib=0.0, pinned_gib=None, ram_tier_gib=0.0):
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
        rt = ram_tier_gib if free >= base + ram_tier_gib else (free - base if free - base >= 4.0 else 0.0)
        if rt < ram_tier_gib: print(f'RAM tier {"shrunk to " + f"{rt:.1f} GiB" if rt else "disabled"}: GTT free {free:.1f} GiB < needed {base + ram_tier_gib:.1f} GiB')
    print(f'host pools: {choice} (GTT limit {"unknown" if lim is None else f"{lim:.1f} GiB"}, free {"unknown" if free is None else f"{free:.1f} GiB"}, '
          f'needed {base:.1f} GiB for GTT pools)' + ('' if choice == 'gtt' else '; user-pointer pools can stall when the kernel compacts host memory'))
    if choice == 'userptr' and pinned_gib:
        busy = '' if free is None or lim is None or lim - free < 1 else (f' ({lim - free:.0f} GiB of the {lim:.0f} GiB GTT limit is in use by other '
                                                                         'processes: stop other GPU servers first if that is not intended)')
        print(f'WARNING host pools: host GTT {"unknown" if free is None else f"{free:.0f} GiB"} < {base:.0f} GiB{busy}: this mode was benchmarked with its '
              f'~{pinned_gib:.0f} GiB of pinned host memory in GTT (by default the GTT limit is half the RAM, so in practice a 256 GB host). Using '
              'pageable pinned (user-pointer) memory instead, without a RAM tier: it starts and runs, but slower than the benchmark (186 vs 192 tok/s, '
              '389 vs 423 tok/s at 8 requests in our validation) and expect GPU stalls of up to several seconds when the kernel compacts host memory. '
              'Raising the GTT limit (amdgpu.gttsize / ttm.pages_limit) is a system setting you may choose; this launcher never changes it.')
    return (dict(hp['env']) if choice == 'gtt' else {}), rt

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

def ram_check(tier_gib, margin_gib=8.0, pinned_gib=None):
    """Pinned host memory of a start: the host tables (every mode) plus the expert host tier (4-bit modes), against MemAvailable."""
    tables = COMMON['host_pools']['gtt_need_gib']; need_gib = max(tables + tier_gib, pinned_gib or 0.0)
    try: avail = int(re.search(r'MemAvailable:\s+(\d+)', open('/proc/meminfo').read()).group(1)) / (1 << 20)
    except Exception: print('ram: /proc/meminfo unreadable; not checked'); return []
    ok = avail >= need_gib + margin_gib
    print(f'ram: pinned host memory {need_gib:.1f} GiB (host tables {tables:g}' + (f' + expert host tier {tier_gib:.1f}' if tier_gib else '') + f', both GPUs); '
          f'available {avail:.1f} GiB; needed {need_gib + margin_gib:.1f} GiB with a {margin_gib:g} GiB margin -> {"ok" if ok else "NOT ENOUGH"}')
    return [] if ok else [f'{need_gib + margin_gib:.0f} GiB of available RAM for the pinned host memory (have {avail:.0f} GiB'
                          + ('; lower the expert tier with --host-pool off' if tier_gib else '') + ')']

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

def ensure_view(weights, view, base_cfg, dry, wanted_optional=()):
    """Runtime view for vLLM (as the benchmarks used): base config with the Paiton quantization block + tokenizer files, and links
    to the container's safetensors as the container will see them (/models/ck/...). Written next to the weights, never inside them."""
    weights, view = pathlib.Path(weights), pathlib.Path(view); ck = COMMON['mounts']['checkpoint']; vdir = COMMON['mounts']['view']
    layers = sorted(p.name for p in (weights / 'layers').glob('L[0-9][0-9].safetensors'))
    links = {n: f'{ck}/layers/{n}' for n in layers}
    links.update({'model.safetensors': f'{ck}/model.safetensors', 'manifest.json': f'{ck}/manifest.json', 'ple-e4m3': f'{ck}/common/ple-e4m3',
                  'mtp-vision.safetensors': f'{ck}/common/bf16-mtp-vision.safetensors', 'mtp-experts.safetensors': f'{ck}/mtp/mtp-experts.safetensors',
                  'mtp-draft-int2/draft-head-int2.safetensors': f'{ck}/mtp/draft-head-int2.safetensors', 'mtp-draft-int2/draft-head-int2.json': f'{ck}/mtp/draft-head-int2.json'})
    for f in sorted((weights / 'trunk-int6' / 'layers').glob('L[0-9][0-9].safetensors')):   # 4-bit INT6 trunk overlay, read as top-level i6-LNN files
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

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='decode', choices=sorted(MODES)); ap.add_argument('--port', type=int, default=18982); ap.add_argument('--name', default='paiton-flashnext')
    ap.add_argument('--weights', default=None, help='weights dir (default per mode: ~/paiton-models/Qwen3.8-Flash-Next-W3A8, 4-bit modes ~/paiton-models/Qwen3.8-Flash-Next-W4A8)');
    ap.add_argument('--expert-vram-gib', type=float, default=None, help='4-bit modes: VRAM per GPU for resident experts (default from modes.json)'); ap.add_argument('--host-pool', choices=('on', 'off'), default=None, help='4-bit modes: keep a pinned host copy of every non-core expert (on, default) or only of the non-resident ones (off)'); ap.add_argument('--cache', default=None, help='compile cache dir (default ~/.cache/paiton/flashnext/<mode>)')
    ap.add_argument('--image', default=None, help='image reference (default: the pinned release image)'); ap.add_argument('--devices', default=None, help='comma list of ROCR ids (default: all AMD GPUs; two are required)')
    ap.add_argument('--view', default=None, help='runtime view dir (default: <weights>-view)'); ap.add_argument('--base-cfg', default=None, help='dir with the base model config + tokenizer files (default: <weights>-base-cfg, downloaded at the pinned revision)')
    ap.add_argument('--prefix-caching', action='store_true', help='opt-in: align-mode prefix caching (recurrent state cached per 2048-token block); see RELEASE-NOTES.md'); ap.add_argument('--served-model-name', default='Qwen3.8-Flash-Next'); ap.add_argument('--host-pools', choices=('auto', 'gtt', 'userptr'), default='auto', help='host-resident tables: driver-owned GTT memory (auto when the GTT size holds them) or ROCm user-pointer memory'); ap.add_argument('--detach', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(); m = MODES[a.mode]
    wspec = COMMON.get(m['weights_key']) if m.get('weights_key') else None
    if a.weights is None: a.weights = os.path.expanduser((wspec or {}).get('default_dir', '~/paiton-models/Qwen3.8-Flash-Next-W3A8'))
    if m.get('status', '').startswith('PENDING') : print(f'mode {a.mode}: {m["status"]}')
    missing = prerequisites(COMMON['tp'])
    tier = None; tier_gib = 0.0
    if m.get('expert_tier'):
        et = m['expert_tier']; tier = tier_spec(m['env'].get(et['env'], et['default']), a.expert_vram_gib, a.host_pool)
        tier_gib = pinned_host_gib(tier, et['expert_bytes_gib_both_ranks'], COMMON['tp'])
    missing += ram_check(tier_gib, 16.0 if m.get('gtt_pinned_gib') else 8.0, m.get('gtt_pinned_gib'))
    if m.get('expert_tier'):
        try: tot = int(re.search(r'MemTotal:\s+(\d+)', open('/proc/meminfo').read()).group(1)) / (1 << 20)
        except Exception: tot = None
        print(f'host RAM: {"unknown" if tot is None else f"{tot:.0f} GiB"} total; the benchmarked 4-bit configuration keeps about 96 GiB of pinned '
              'host memory in one GPU\'s GTT, so it needs about 98 GiB of free GTT (the GTT limit is half the RAM by default: a 256 GB host); '
              'see the host pools line below')
    if missing and not a.dry_run: sys.exit('fix the missing prerequisites above and run again')
    for rel in m.get('requires', []):
        if not (pathlib.Path(a.weights) / rel).exists() and not a.dry_run: sys.exit(f'mode {a.mode} needs {rel} in the checkpoint dir (not in this copy of the repo)')
    gpus = a.devices.split(',') if a.devices else amd_gpus()
    if len(gpus) < COMMON['tp'] and not a.dry_run: sys.exit(f'need {COMMON["tp"]} AMD GPUs, found {gpus}')
    gpus = gpus[:COMMON['tp']]
    p2p_check(gpus)
    ispec = COMMON[m.get('image_key', 'image')]
    image = a.image or (ispec['tag'] + ('@' + ispec['digest'] if ispec['digest'] != 'PIN-ME' else ''))
    weights = ensure_weights(a.weights, a.dry_run, *((wspec['hf_repo'], wspec['hf_revision']) if wspec else (None, None)))
    want_rel = (wspec or COMMON)['hf_repo'].split('/')[-1]
    try: got_rel = json.loads((weights / 'config.json').read_text()).get('paiton_release')
    except Exception: got_rel = None
    if got_rel and got_rel != want_rel:
        msg = f'mode {a.mode} expects the {want_rel} weights, but {weights} holds {got_rel}; pass --weights with the matching download'
        if a.dry_run: print('WARNING:', msg)
        else: sys.exit(msg)
    base_cfg = ensure_base_cfg(a.base_cfg or str(weights) + '-base-cfg', a.dry_run); view = ensure_view(weights, a.view or str(weights) + '-view', base_cfg, a.dry_run, tuple(m.get('requires', [])))
    cache = pathlib.Path(a.cache or os.path.expanduser(f'~/.cache/paiton/flashnext/{a.mode}')); cache.mkdir(parents=True, exist_ok=True)
    env = dict(m['env']); env['VLLM_CACHE_ROOT'] = COMMON['mounts']['cache']; env['HF_HUB_OFFLINE'] = '1'
    if tier: env[m['expert_tier']['env']] = tier
    engine_args = list(m['engine_args'])
    if a.prefix_caching:
        pc = COMMON['prefix_caching']; engine_args = [x for x in engine_args if x not in pc['engine_args_drop']] + pc['engine_args_add']; env.update(pc['env'])
    env['ROCR_VISIBLE_DEVICES'] = ','.join(gpus); env['HIP_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(len(gpus))); env['CUDA_VISIBLE_DEVICES'] = env['HIP_VISIBLE_DEVICES']
    env['HOME'] = '/tmp'; env['USER'] = env['LOGNAME'] = 'paiton'
    for k in COMMON['host_pools']['env']: env.pop(k, None)   # a mode profile may carry it; the host-pools choice decides
    hp_env, _ = host_pool_env(a.host_pools, gpus, tier_gib, pinned_gib=m.get('gtt_pinned_gib'))
    env.update(hp_env)
    env['PAITON_FN_PLACEMENT'] = f'counts:{COMMON["mounts"]["checkpoint"]}/common/routing-counts.json'   # the container runs as the host uid, which has no passwd entry: getpass.getuser() needs these
    cmd = ['docker', 'run', '--rm', '--name', a.name, '--user', f'{os.getuid()}:{os.getgid()}'] + COMMON['docker']
    gid = render_gid(); cmd += ['--group-add', gid] if gid else []
    if a.detach: cmd += ['-d']
    for k, v in sorted(env.items()): cmd += ['-e', f'{k}={v}']
    cmd += ['-v', f'{weights}:{COMMON["mounts"]["checkpoint"]}:ro', '-v', f'{view}:{COMMON["mounts"]["view"]}:ro', '-v', f'{cache}:{COMMON["mounts"]["cache"]}']
    # first start of a mode: copy the baked compile-cache seed into the (empty) cache, then serve
    inner = (f'if [ -d /opt/paiton/cache-seed/{m.get("seed", a.mode)} ] && [ -z "$(ls -A {COMMON["mounts"]["cache"]} 2>/dev/null)" ]; then cp -a /opt/paiton/cache-seed/{m.get("seed", a.mode)}/. {COMMON["mounts"]["cache"]}/; fi; '
             f'exec python3 -m vllm.entrypoints.openai.api_server --model {COMMON["mounts"]["view"]} --tokenizer {COMMON["mounts"]["view"]} '
             + ' '.join(shlex.quote(x) for x in engine_args) + f' --served-model-name {shlex.quote(a.served_model_name)} --host 0.0.0.0 --port {a.port}')
    cmd += ['--entrypoint', 'bash', image, '-c', inner]
    print(f'mode {a.mode}: image {image}; GPUs {gpus}; weights {weights}; cache {cache}; port {a.port}' + (f'; expert tier {tier}' if tier else ''))
    if a.dry_run: print(' '.join(shlex.quote(x) for x in cmd)); return
    os.execvp('docker', cmd)

if __name__ == '__main__':
    main()
