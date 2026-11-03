"""--extend-cache (session hibernation v1): the native cache modes size the prefix-cache tiers behind VRAM from the host:
system memory when its tier holds >= 1.25x the GPU pool, else an NVMe/SSD disk tier (NativeTieringOffloadingSpec +
NativeFsTier, sha256 block hashes, a 0700 fingerprint folder mounted at /kvdisk) behind a RAM staging tier of one session;
refusals: spinning disk, too little free space, too little RAM for the staging tier, the root directory, non-native modes,
--native-kv-tier with ram / auto; --wipe-disk-cache removes only the launcher's folders."""
import json
import pathlib
import shlex
import stat
import sys

import pytest
from test_flashnext_native_launch import dry, lf  # noqa: F401  (lf: the launcher module fixture)


def kvt_cfg(inner):
    eng = shlex.split(inner.split('exec ', 1)[1])
    return json.loads(eng[eng.index('--kv-transfer-config') + 1]), eng


def host(lf, monkeypatch, avail, rotational=False, free_gib=1000.0):
    monkeypatch.setattr(lf, 'meminfo_gib', lambda f: avail)
    monkeypatch.setattr(lf, 'storage_is_rotational', lambda p: rotational)
    monkeypatch.setattr(lf.shutil, 'disk_usage', lambda p: type('U', (), {'free': free_gib * 2 ** 30})())


def test_auto_picks_ram_when_it_holds_the_pool(lf, monkeypatch, tmp_path, capsys):
    host(lf, monkeypatch, 200.0)
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'native', '--prefix-caching', '--extend-cache',
                                      '--disk-cache-dir', str(tmp_path / 'kv'), mode='decode')
    cfg, eng = kvt_cfg(inner)
    assert '--extend-cache auto: system memory, 16 GiB per GPU' in out
    assert cfg['kv_connector_extra_config']['spec_name'] == 'NativeCPUOffloadingSpec' and '--prefix-caching-hash-algo' not in eng
    assert not any(v.endswith(':/kvdisk:rw') for v in vols)


def test_auto_small_host_gets_the_disk_tier(lf, monkeypatch, tmp_path, capsys):
    host(lf, monkeypatch, 64.0)   # 3-bit cache mode on a 64 GiB host: the RAM tier is off, the disk tier fits a 1 GiB staging tier
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--extend-cache', '--disk-cache-dir', str(tmp_path / 'kv'),
                                      '--memory', '64', mode='3bit-cache-native')   # what --prefix-caching routes a 64 GiB host to
    cfg, eng = kvt_cfg(inner)
    x = cfg['kv_connector_extra_config']
    assert x['spec_name'] == 'NativeTieringOffloadingSpec' and x['cpu_bytes_to_use'] == 2 << 30
    assert x['secondary_tiers'] == [{'type': 'NativeFsTier', 'module_path': 'paiton_vllm_plugin.flashnext.fnkvt', 'root_dir': '/kvdisk',
                                     'max_bytes': 64 << 30}]
    assert eng[eng.index('--prefix-caching-hash-algo') + 1] == 'sha256' and env['PAITON_FN_KVT_HOST_RESERVE_GIB'] == '3'
    assert 'behind a 1 GiB per GPU RAM tier (one session needs 1)' in out and 'up to 64 GiB' in out
    disk = [v for v in vols if v.endswith(':/kvdisk:rw')]
    assert len(disk) == 1 and disk[0].startswith(str(tmp_path / 'kv' / 'flashnext-'))


def test_disk_tier_folder_fingerprint(lf, monkeypatch, tmp_path, capsys):
    host(lf, monkeypatch, 240.0)
    lib = tmp_path / 'lib' / 'libpaiton_fnexec.so'; lib.parent.mkdir(); lib.write_bytes(b'a')
    monkeypatch.setattr(sys, 'argv', [])

    def run(*extra):
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--extend-cache', 'disk',
                                          '--disk-cache-dir', str(tmp_path / 'kv'), *extra, mode='4bit-long-cache-mtp-native')
        return next(v for v in vols if v.endswith(':/kvdisk:rw')).split(':')[0], out
    a, out = run()
    assert 'behind a 16 GiB per GPU RAM tier (one session needs 2.5)' in out   # a 262K session: 142 slots of 15.76 MB
    b, _ = run()
    lib.write_bytes(b'b')   # another executor build: another folder (its records may differ)
    c, _ = run()
    assert a == b != c
    d, out = run('--native-kv-tier', '10')   # the RAM tier in front of the disk, explicit
    assert 'behind a 10 GiB per GPU RAM tier' in out and d == c
    with pytest.raises(SystemExit, match='at least 2.5 GiB per GPU'):
        run('--native-kv-tier', '1')


def test_real_start_creates_a_private_folder(lf, monkeypatch, tmp_path):
    host(lf, monkeypatch, 240.0)
    a = type('A', (), {})(); a.dry_run = False; a.mode = '4bit-long-cache-mtp-native'; a.drafter = None; a.spec_tokens = None
    a.disk_tier = {'base': tmp_path / 'kv', 'cap_gib': 1e-6, 'reserve_gib': 8}
    monkeypatch.setattr(lf.subprocess, 'run', lambda *x, **k: type('R', (), {'stdout': 'sha256:img'})())
    f = lf.disk_cache_folder(a, lf.MODES[a.mode], 'img', None, tmp_path, 262144)
    assert stat.S_IMODE(f.stat().st_mode) & 0o777 == 0o700 and json.loads((f / 'FINGERPRINT.json').read_text())['image'] == 'sha256:img'
    (f / 'x.bin').write_bytes(b'0' * 4096)   # over its (tiny) cap: removed at the next start, then recreated empty
    g = lf.disk_cache_folder(a, lf.MODES[a.mode], 'img', None, tmp_path, 262144)
    assert g == f and not (f / 'x.bin').exists()


@pytest.mark.parametrize('case', ['hdd', 'full', 'ram', 'root', 'mode', 'tier'])
def test_refusals(lf, monkeypatch, tmp_path, capsys, case):
    host(lf, monkeypatch, 60.0 if case == 'ram' else 240.0, rotational=case == 'hdd', free_gib=6.0 if case == 'full' else 1000.0)
    extra = ['--disk-cache-dir', '/' if case == 'root' else str(tmp_path / 'kv'), '--extend-cache', 'disk']
    mode = '3bit-cache-native'
    if case == 'mode':
        mode = '3bit-mtp-native'
    if case == 'tier':
        extra = ['--disk-cache-dir', str(tmp_path / 'kv'), '--extend-cache', 'ram', '--native-kv-tier', '4']
    if case == 'ram':
        extra += ['--memory', '60']
    want = {'hdd': 'spinning disk', 'full': 'holds less than one session', 'ram': 'this host has room for less',
            'root': 'not the root directory', 'mode': 'has no native prefix-cache tier', 'tier': 'drop --native-kv-tier'}[case]
    with pytest.raises(SystemExit, match=want):
        dry(lf, monkeypatch, tmp_path, capsys, *extra, mode=mode)


def test_wipe_removes_only_launcher_folders(lf, monkeypatch, tmp_path, capsys):
    base = tmp_path / 'kv'
    for n in ('flashnext-aaaa', 'flashnext-bbbb', 'qwen38-cccc', 'mine'):
        (base / n).mkdir(parents=True); (base / n / 'f.bin').write_bytes(b'x')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--wipe-disk-cache', '--disk-cache-dir', str(base)])
    lf.main()
    assert sorted(p.name for p in base.iterdir()) == ['mine', 'qwen38-cccc']
    assert '2 folder(s)' in capsys.readouterr().out
