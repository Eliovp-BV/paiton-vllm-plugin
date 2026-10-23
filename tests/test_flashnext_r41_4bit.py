"""r4.1 --mode 4bit launcher defaults: expert tier 21.4 + rotary cache trim on for text; --vision keeps the full rotary cache and
takes its own lower tier (21.2); PAITON_FN_ROPE_TRIM in the caller's environment overrides the mode default."""
import importlib.util
import pathlib
import re
import sys

import pytest
from conftest import fixed_host_open

LAUNCHER = pathlib.Path(__file__).resolve().parents[1] / 'models' / 'Qwen3.8-Flash-Next' / 'launch-flashnext.py'


@pytest.fixture()
def lf():
    spec = importlib.util.spec_from_file_location('launch_flashnext', LAUNCHER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.open = fixed_host_open()   # fixed host RAM (conftest), not the live /proc/meminfo
    return mod


def dry(lf, monkeypatch, tmp_path, capsys, *extra):
    monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', lambda *a, **k: [])
    monkeypatch.setattr(lf, 'render_gid', lambda: None); monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (200.0, 200.0))
    monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w')
    monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
    monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', '4bit', '--dry-run', '--devices', 'GPU-0,GPU-1',
                                      '--cache', str(tmp_path / 'cache'), *extra])
    lf.main()
    out = capsys.readouterr().out
    return out, re.search(r'-e PAITON_FN_EXPERT_TIER=static:vram_gib=([0-9.]+),', out).group(1), re.search(r'-e PAITON_FN_ROPE_TRIM=(\d)', out)


def test_text_defaults(lf, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv('PAITON_FN_ROPE_TRIM', raising=False)
    _, tier, rope = dry(lf, monkeypatch, tmp_path, capsys)
    assert tier == '20.75' and rope is None   # r6: the torch text default = the published r3 configuration


def test_vision_own_tier_no_rope_trim(lf, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv('PAITON_FN_ROPE_TRIM', raising=False)
    out, tier, rope = dry(lf, monkeypatch, tmp_path, capsys, '--vision', '--image', 'flashnext-test:vision')
    assert tier == '21.2' and rope is None and 'image input: expert VRAM 21.2 GiB per GPU (text-only 20.75)' in out


def test_explicit_expert_vram_wins_over_vision_tier(lf, monkeypatch, tmp_path, capsys):
    _, tier, _ = dry(lf, monkeypatch, tmp_path, capsys, '--vision', '--image', 'flashnext-test:vision', '--expert-vram-gib', '20.5')
    assert tier == '20.5'


def test_env_override_turns_rope_trim_off(lf, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('PAITON_FN_ROPE_TRIM', '0')
    _, _, rope = dry(lf, monkeypatch, tmp_path, capsys)
    assert rope and rope.group(1) == '0'


def test_other_4bit_modes_unchanged(lf):
    for mode in ('4bit-200k', '4bit-cache', '4bit-long', '4bit-xl'):
        assert 'PAITON_FN_ROPE_TRIM' not in lf.MODES[mode]['env'] and 'vram_gib' not in (lf.MODES[mode].get('vision') or {})
