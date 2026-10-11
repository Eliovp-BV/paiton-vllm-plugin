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
