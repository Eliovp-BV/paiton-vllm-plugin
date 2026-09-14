"""Download hash-pinned build inputs; preserve existing files and verify reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    lock = Path(__file__).resolve().parents[1] / 'media-sources.lock.json'
    rows = json.loads(lock.read_text())
    for row in rows:
        target = destination / row['filename']
        if target.exists():
            if digest(target) != row['sha256']:
                raise ValueError(f'{target.name}: existing file hash differs; preserved.')
            continue
        fd, name = tempfile.mkstemp(prefix='.download-', dir=destination)
        temporary = Path(name)
        try:
            with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(row['url'], timeout=120) as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
            if digest(temporary) != row['sha256']:
                raise ValueError(f'{target.name}: download hash mismatch.')
            # Do not overwrite another preparer's completed file.
            try:
                os.link(temporary, target)
            except FileExistsError:
                if digest(target) != row['sha256']:
                    raise ValueError(f'{target.name}: concurrent file hash differs; preserved.')
        finally:
            temporary.unlink(missing_ok=True)
    print(f'Verified {len(rows)} pinned media build inputs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args().output)
