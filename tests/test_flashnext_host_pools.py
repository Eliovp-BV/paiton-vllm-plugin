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
