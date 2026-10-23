import pytest


@pytest.fixture(autouse=True)
def _no_image_verify(monkeypatch):
    # launch-flashnext.py --dry-run checks the baked drafter files inside a local image with docker; tests never start containers
    monkeypatch.setenv('FLASHNEXT_DRY_SKIP_IMAGE_VERIFY', '1')


# The launcher sizes host pools, the RAM checks and the RAM prefix tier from /proc/meminfo. Tests read a fixed idle 256 GB
# host instead (MemTotal 251.6, MemAvailable 240 GiB), so results do not depend on the box or on a server running beside
# them; a test that needs another host patches the launcher's open / meminfo_gib itself.
FIXED_MEMINFO = f'MemTotal: {int(251.6 * (1 << 20))} kB\nMemAvailable: {int(240.0 * (1 << 20))} kB\n'


def fixed_host_open(real_open=open):
    import io

    def _open(path, *a, **k):
        return io.StringIO(FIXED_MEMINFO) if str(path) == '/proc/meminfo' else real_open(path, *a, **k)
    return _open
