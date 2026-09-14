"""Preserve the pinned ROCm runtime's hard-link groups across image layers."""
import argparse
import json
import os
from pathlib import Path
import stat


def snapshot(root):
    groups = {}
    for directory, _, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                groups.setdefault((info.st_dev, info.st_ino), []).append(str(path))
    return {'schema': 1, 'root': str(root), 'groups': sorted(
        sorted(paths) for paths in groups.values() if len(paths) > 1)}


def check(manifest):
    broken = []
    for paths in manifest['groups']:
        identities = {(p.stat().st_dev, p.stat().st_ino) for p in map(Path, paths)}
        if len(identities) != 1:
            broken.append(paths)
    if broken:
        raise RuntimeError(f'{len(broken)} runtime hard-link groups were split; first: {broken[0]}')
    print(f"Verified {len(manifest['groups'])} runtime hard-link groups")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['snapshot', 'check'])
    parser.add_argument('--root', type=Path, default=Path('/opt/python'))
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'snapshot':
        args.manifest.write_text(json.dumps(snapshot(args.root), indent=2))
    else:
        check(json.loads(args.manifest.read_text()))


if __name__ == '__main__':
    main()
