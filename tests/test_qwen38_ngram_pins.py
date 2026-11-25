"""The n-gram co-drafting policy in the pinned 27B image must pin the sources that image executes.

The adapter (PAITON_NGRAM_CODRAFT=1) refuses to arm when a pinned vLLM source differs from the one it runs, so a pin
left behind after an overlay change turns the documented switch off without an error (images r1 to r3). This test
reads the policy and the sources out of the image the launcher pins and compares every pin the adapter checks.
It needs docker and the image present locally; PAITON_REQUIRE_IMAGE_TESTS=1 (release checks) turns the skip into a
failure. PAITON_NGRAM_PIN_IMAGE checks another image, e.g. a local candidate.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'models' / 'Qwen3.8-MXFP4-DFlash2' / 'launch-rocm10.py'
SPEC = importlib.util.spec_from_file_location('rocm10_launcher_ngram_pins', SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

# Runs inside the image with python3 -I -S: no site packages, so no .pth hook and no import of the policy.
CHECK = r'''
import ast, hashlib, json, sysconfig
from pathlib import Path
DP = Path(sysconfig.get_paths()['purelib'])   # python3.12 on the vLLM 0.29 images, python3.14 on 0.32
POLICY = DP / 'paiton_ngram_codraft_policy.py'
if not POLICY.exists():
    print(json.dumps({'absent': True})); raise SystemExit
GROUPS = ('PATCHED', 'DEPENDENCIES', 'SOURCE_CAP_PATCHED', 'SOURCE_CAP_DEPENDENCIES', 'PREFILL_SKIP_SHA')
pins = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(POLICY.read_text()).body
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and getattr(n.targets[0], 'id', None) in GROUPS}
files = json.loads((DP / 'paiton_runtime_compat' / 'compat_manifest.json').read_text())['files']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = []
for group in GROUPS[:-1]:
    for name, digest in pins.get(group, {}).items():
        original = DP / (name.replace('.', '/') + '.py')
        entry = files.get(str(original))   # the overlay loader executes the overlay payload when the manifest lists one
        path = DP / 'paiton_runtime_compat' / entry['overlay'] if entry else original
        rows.append([group, name, digest, sha(path) if path.exists() else None])
rows.append(['PREFILL_SKIP_SHA', 'paiton_dflash_prefill_skip_adapter', pins.get('PREFILL_SKIP_SHA'),
             sha(DP / 'paiton_dflash_prefill_skip_adapter.py')])
print(json.dumps({'absent': False, 'rows': rows}))
'''


# Released images whose stale pins are known and documented (REFERENCE.md known issue). Exactly these rows make the
# test an expected failure; any other difference still fails. Remove the entry when the launcher pins a fixed image.
KNOWN_STALE = {}


def pinned_image():
    return os.environ.get('PAITON_NGRAM_PIN_IMAGE') or launcher.IMAGES['65k']


def unavailable(reason):
    if os.environ.get('PAITON_REQUIRE_IMAGE_TESTS') == '1':
        pytest.fail(reason)
    pytest.skip(reason)


def test_ngram_policy_pins_match_the_pinned_image():
    image = pinned_image()
    if image.partition('@')[0].endswith(launcher.UPSTREAM_NO_DROP_IMAGE_SUFFIXES):
        # the vLLM 0.32 images carry the 0.29 adapter unported; the launcher does not pass the switch on and says so
        # (test_qwen38_rocm10_launcher: test_ngram_codraft_forwarded_only_when_set_on_host)
        pytest.xfail(f'{image}: n-gram co-drafting is not offered on the vLLM 0.32 images')
    if shutil.which('docker') is None:
        unavailable('docker is not installed')
    if subprocess.run(['docker', 'image', 'inspect', image], capture_output=True).returncode != 0:
        unavailable(f'{image} is not present locally')
    result = subprocess.run(['docker', 'run', '--rm', '--name', f'paiton-ngram-pin-test-{os.getpid()}', '--network', 'none',
                             '--entrypoint', 'python3', image, '-I', '-S', '-c', CHECK],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert not report['absent'], f'{image} ships no n-gram policy, but the docs offer PAITON_NGRAM_CODRAFT=1'
    stale = [f'{group} {name}: pinned {pinned[:12]}, image runs {(actual or "missing")[:12]}'
             for group, name, pinned, actual in report['rows'] if pinned != actual]
    known = KNOWN_STALE.get(image.partition('@')[0].rpartition(':')[2])
    if stale and known is not None and set(stale) == known:
        pytest.xfail(f'known issue of {image}: the n-gram switch does not arm (fixed in the next image)')
    assert not stale, f'n-gram co-drafting cannot arm on {image}; review the source diff, then update the pins:\n' + \
        '\n'.join(stale)
