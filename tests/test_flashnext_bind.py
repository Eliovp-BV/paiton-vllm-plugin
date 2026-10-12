"""Flash Next launcher: the API listens on every interface by default; --host keeps it local; --api-key is opt-in."""
import importlib.util
import pathlib
import shlex
import sys

import pytest

LAUNCHER = pathlib.Path(__file__).resolve().parents[1] / 'models' / 'Qwen3.8-Flash-Next' / 'launch-flashnext.py'


@pytest.fixture()
def lf():
    spec = importlib.util.spec_from_file_location('launch_flashnext_bind', LAUNCHER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def dry_run(lf, monkeypatch, capsys, tmp_path, *extra):
    monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', lambda *a, **k: [])
    monkeypatch.setattr(lf, 'render_gid', lambda: None); monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (125.8, 125.8))
    monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w'); monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
    monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--dry-run', '--devices', 'GPU-0,GPU-1', '--cache', str(tmp_path / 'cache'), *extra])
    lf.main()
    out = capsys.readouterr().out
    argv = shlex.split(next(line for line in out.splitlines() if line.startswith('docker run')))
    inner = shlex.split(argv[-1])
    return out, argv, inner


@pytest.mark.parametrize('mode', ['decode', '4bit'])
def test_api_listens_on_every_interface_by_default(lf, monkeypatch, capsys, tmp_path, mode):
    out, argv, inner = dry_run(lf, monkeypatch, capsys, tmp_path, '--mode', mode)
    assert argv[argv.index('--network') + 1] == 'host' and inner[inner.index('--host') + 1] == '0.0.0.0'
    assert 'VLLM_API_KEY' not in out


def test_host_and_api_key_are_opt_in(lf, monkeypatch, capsys, tmp_path):
    out, argv, inner = dry_run(lf, monkeypatch, capsys, tmp_path, '--host', '127.0.0.1', '--api-key', 'k-test-123')
    assert inner[inner.index('--host') + 1] == '127.0.0.1'
    assert argv[argv.index('VLLM_API_KEY') - 1] == '-e' and 'k-test-123' not in out


@pytest.mark.parametrize('host,probe,remote', [('0.0.0.0', '127.0.0.1', True), ('127.0.0.1', '127.0.0.1', False),
                                               ('192.0.2.7', '192.0.2.7', False)])
def test_readiness_probe_follows_the_bind(lf, monkeypatch, capsys, host, probe, remote):
    import urllib.request
    seen = []

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, 'urlopen', lambda url, timeout: seen.append(url) or Response())
    assert lf.wait_ready(18982, None, 'x', settle_s=0, drop_cache=False, host=host)
    out = capsys.readouterr().out
    assert seen == [f'http://{probe}:18982/health'] and f'API http://{probe}:18982/v1' in out
    assert ('<this-machine-ip>' in out) == remote
