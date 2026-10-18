import pytest


@pytest.fixture(autouse=True)
def _no_image_verify(monkeypatch):
    # launch-flashnext.py --dry-run checks the baked drafter files inside a local image with docker; tests never start containers
    monkeypatch.setenv('FLASHNEXT_DRY_SKIP_IMAGE_VERIFY', '1')
