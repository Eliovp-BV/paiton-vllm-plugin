"""Host-pool ladder of the Flash Next launcher (4-bit modes: GTT pools only when the free GTT holds the measured pinned memory)."""
import importlib.util
import pathlib

import pytest

LAUNCHER = pathlib.Path(__file__).resolve().parents[1] / 'models' / 'Qwen3.8-Flash-Next' / 'launch-flashnext.py'


@pytest.fixture()
def lf():
    spec = importlib.util.spec_from_file_location('launch_flashnext', LAUNCHER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def run(lf, monkeypatch, free, choice='auto', pinned=95.5, ram_tier=0.0):
    monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (None, None) if free is None else (free, free))
    return lf.host_pool_env(choice, ['GPU-0'], 40.8, pinned_gib=pinned, ram_tier_gib=ram_tier)


def test_gtt_when_pinned_fits(lf, monkeypatch):
    assert run(lf, monkeypatch, 125.8) == ({'HSA_USERPTR_FOR_PAGED_MEM': '0'}, 0.0)


def test_userptr_fallback_on_200gb_host(lf, monkeypatch, capsys):
    assert run(lf, monkeypatch, 93.0) == ({}, 0.0)
    assert 'WARNING host pools' in capsys.readouterr().out


def test_unknown_gtt_falls_back(lf, monkeypatch):
    assert run(lf, monkeypatch, None) == ({}, 0.0)


def test_ram_tier_full_shrunk_off(lf, monkeypatch):
    assert run(lf, monkeypatch, 125.8, ram_tier=12.0)[1] == 12.0
    assert run(lf, monkeypatch, 105.0, ram_tier=12.0)[1] == pytest.approx(7.5)
    assert run(lf, monkeypatch, 99.0, ram_tier=12.0) == ({'HSA_USERPTR_FOR_PAGED_MEM': '0'}, 0.0)
    assert run(lf, monkeypatch, 90.0, ram_tier=12.0) == ({}, 0.0)


def test_explicit_gtt_refuses_when_short(lf, monkeypatch):
    with pytest.raises(SystemExit):
        run(lf, monkeypatch, 93.0, choice='gtt')


def test_3bit_rule_unchanged(lf, monkeypatch):
    monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (62.0, 62.0))
    assert lf.host_pool_env('auto', ['GPU-0'])[0] == {'HSA_USERPTR_FOR_PAGED_MEM': '0'}
    monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (58.0, 58.0))
    assert lf.host_pool_env('auto', ['GPU-0'])[0] == {}


def _health_server():
    import http.server, threading
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self): self.send_response(200 if self.path == '/health' else 404); self.end_headers()
        def log_message(self, *a): pass
    srv = http.server.HTTPServer(('127.0.0.1', 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_wait_ready_settles_after_migration_burst(lf, monkeypatch, tmp_path, capsys):
    import subprocess as sp
    srv = _health_server(); proc = sp.Popen(['sleep', '30'])
    seq = iter([0] + [i * 20000 for i in range(1, 7)] + [120000 + i * 100 for i in range(1, 40)])   # 3 s of 40K pages/s, then quiet
    monkeypatch.setattr(lf, 'pgmigrate_success', lambda: next(seq))
    (tmp_path / 'a.safetensors').write_bytes(b'x' * 4096)
    try:
        assert lf.wait_ready(srv.server_address[1], proc, 'x', settle_s=12, quiet_s=2, weights=tmp_path)
    finally:
        proc.kill(); srv.shutdown()
    out = capsys.readouterr().out
    assert 'dropped the page cache of 1 checkpoint files' in out and 'READY on port' in out and 'settle wait' in out


def test_wait_ready_false_when_server_exits(lf):
    import subprocess as sp
    proc = sp.Popen(['true']); proc.wait()
    assert lf.wait_ready(1, proc, 'x', settle_s=0, weights=None) is False


def dry_cmd(lf, monkeypatch, tmp_path, capsys, kv_tier):
    """main() --dry-run for --mode 4bit with the host checks stubbed; returns the printed docker command."""
    import copy, sys
    modes = copy.deepcopy(lf.MODES)
    if kv_tier: modes['4bit']['env']['PAITON_FN_KV_TIER'] = '1'
    monkeypatch.setattr(lf, 'MODES', modes)
    monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', lambda *a, **k: [])
    monkeypatch.setattr(lf, 'render_gid', lambda: None)
    monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (200.0, 200.0))
    monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w')
    monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
    monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', '4bit', '--dry-run', '--devices', 'GPU-0,GPU-1', '--cache', str(tmp_path / 'cache')])
    lf.main()
    return capsys.readouterr().out


def test_kv_tier_hook_mounted_only_with_kv_tier(lf, monkeypatch, tmp_path, capsys):
    target = '/usr/local/lib/python3.12/dist-packages/zz_paiton_fn_kv_tier.pth:ro'
    assert target not in dry_cmd(lf, monkeypatch, tmp_path, capsys, False)
    out = dry_cmd(lf, monkeypatch, tmp_path, capsys, True)
    assert f'{tmp_path}/cache-kv-tier.pth:{target}' in out
    assert "PAITON_FN_KV_TIER', '0') == '1'" in lf.KV_TIER_PTH   # the hook stays a no-op without the env


def test_int6_overlay_linked_only_for_int6_modes(lf, tmp_path):
    w = tmp_path / 'w'; (w / 'layers').mkdir(parents=True); (w / 'trunk-int6' / 'layers').mkdir(parents=True); (w / 'common').mkdir(); (w / 'mtp').mkdir()
    for n in ('L00', 'L01'):
        (w / 'layers' / f'{n}.safetensors').write_bytes(b'x'); (w / 'trunk-int6' / 'layers' / f'{n}.safetensors').write_bytes(b'x')
    for rel in ('model.safetensors', 'manifest.json', 'common/ple-e4m3', 'common/bf16-mtp-vision.safetensors', 'mtp/mtp-experts.safetensors',
                'mtp/draft-head-int2.safetensors', 'mtp/draft-head-int2.json'):
        (w / rel).write_bytes(b'x')
    cfg = tmp_path / 'cfg'; cfg.mkdir(); (cfg / 'config.json').write_text('{}')
    plain = lf.ensure_view(w, tmp_path / 'view', cfg, False)
    assert not [p for p in plain.iterdir() if p.name.startswith('i6-')]
    with_int6 = lf.ensure_view(w, tmp_path / 'view', cfg, False, int6=True)   # same dir: rebuilt with the overlay links
    assert sorted(p.name for p in with_int6.iterdir() if p.name.startswith('i6-')) == ['i6-L00.safetensors', 'i6-L01.safetensors']
    again = lf.ensure_view(w, tmp_path / 'view', cfg, False)                  # and back: stale overlay links removed
    assert not [p for p in again.iterdir() if p.name.startswith('i6-')]


def cache_cmd(lf, monkeypatch, capsys, tmp_path, free, extra=()):
    """main() --dry-run for --mode 4bit-cache with the host checks stubbed and `free` GiB of free GTT."""
    import sys
    monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', lambda *a, **k: [])
    monkeypatch.setattr(lf, 'render_gid', lambda: None); monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (free, free))
    monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w'); monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
    monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', '4bit-cache', '--dry-run', '--devices', 'GPU-0,GPU-1', '--cache', str(tmp_path / 'cache'), *extra])
    lf.main()
    return capsys.readouterr().out


def test_cache_mode_ram_tier_ladder(lf, monkeypatch, capsys, tmp_path):
    hook = 'zz_paiton_fn_kv_tier.pth:ro'
    full = cache_cmd(lf, monkeypatch, capsys, tmp_path, 125.8)
    assert '"cpu_bytes_to_use": 8589934592' in full and 'PAITON_FN_KV_TIER=1' in full and 'PAITON_HOST_KV_PRIVATE=1' in full and hook in full
    assert '--enable-prefix-caching --block-size 2048 --mamba-cache-mode align' in full and '--no-enable-prefix-caching' not in full
    assert 'PAITON_FN_KV_LEASE=off' in full and 'VLLM_USE_V2_MODEL_RUNNER' not in full
    part = cache_cmd(lf, monkeypatch, capsys, tmp_path, 103.0)          # 5.5 GiB beyond pinned + margin: 5 GiB tier
    assert '"cpu_bytes_to_use": 5368709120' in part
    off = cache_cmd(lf, monkeypatch, capsys, tmp_path, 99.0)            # GTT pools fit, the tier does not
    assert 'WARNING RAM prefix tier: off' in off and 'kv-transfer-config' not in off and 'PAITON_FN_KV_TIER' not in off and hook not in off
    assert 'PAITON_HOST_KV_PRIVATE' not in off and '--enable-prefix-caching' in off
    up = cache_cmd(lf, monkeypatch, capsys, tmp_path, 90.0)             # user-pointer pools: no tier
    assert 'kv-transfer-config' not in up and 'HSA_USERPTR_FOR_PAGED_MEM' not in up.split('docker run')[1]


def test_cache_mode_ignores_prefix_caching_flag(lf, monkeypatch, capsys, tmp_path):
    out = cache_cmd(lf, monkeypatch, capsys, tmp_path, 125.8, ('--prefix-caching',))
    assert 'prefix caching is part of this mode' in out and 'VLLM_USE_V2_MODEL_RUNNER' not in out
    assert out.count('--enable-prefix-caching') == 1 and '--mamba-cache-mode align' in out


def mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, free, extra=()):
    import sys
    monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', lambda *a, **k: [])
    monkeypatch.setattr(lf, 'render_gid', lambda: None); monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (free, free))
    monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w'); monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
    monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', mode, '--dry-run', '--devices', 'GPU-0,GPU-1', '--cache', str(tmp_path / 'cache'), *extra])
    lf.main()
    return capsys.readouterr().out


@pytest.mark.parametrize('mode', ['decode', 'prefill-long', 'decode-nopf', 'prefill-long-nopf'])
def test_3bit_modes_default_userptr_gtt_opt_in(lf, monkeypatch, capsys, tmp_path, mode):
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8)
    assert 'HSA_USERPTR_FOR_PAGED_MEM' not in out.split('docker run')[1] and '--host-pools gtt removes' in out
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, ('--host-pools', 'gtt'))
    assert 'HSA_USERPTR_FOR_PAGED_MEM=0' in out.split('docker run')[1]


def test_4bit_default_stays_gtt(lf, monkeypatch, capsys, tmp_path):
    assert 'HSA_USERPTR_FOR_PAGED_MEM=0' in mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 125.8).split('docker run')[1]


@pytest.mark.parametrize('mode', ['4bit', '4bit-200k'])
def test_prefix_caching_refused_on_unmeasured_4bit_modes(lf, monkeypatch, capsys, tmp_path, mode):
    with pytest.raises(SystemExit, match='use --mode 4bit-cache'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, ('--prefix-caching',))
