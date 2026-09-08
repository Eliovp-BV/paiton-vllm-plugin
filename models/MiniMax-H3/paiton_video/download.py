"""Fetch immutable upstream checkpoints and verify their full SHA-256 digests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import tempfile

from huggingface_hub import hf_hub_download

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile",choices=["base20","turbo8","turbo4","all"],default="turbo8")
parser.add_argument("--verify-only",action="store_true")
parser.add_argument("--rehash",action="store_true")
args=parser.parse_args()
root=Path(os.environ.get("PAITON_H3_DATA","/data")).resolve()
lock=json.loads((Path(__file__).resolve().parents[1]/"checkpoints.lock.json").read_text())
verified=root/"cache/verified"
verified.mkdir(parents=True,exist_ok=True)
for item in lock["files"]:
    if args.profile!="all" and args.profile not in item["profiles"]:
        continue
    target=root/"models"/item["destination"]
    if not target.exists():
        if args.verify_only:
            raise FileNotFoundError(target.name)
        cached=Path(hf_hub_download(item["repository"],item["file"],revision=item["revision"],
                                   cache_dir=root/"cache/hub")).resolve()
        target.parent.mkdir(parents=True,exist_ok=True)
        # Each attempt owns its temporary path; an interrupted previous
        # download must not make the next launch copy onto a cache hard link.
        with tempfile.TemporaryDirectory(prefix=".h3-download-",dir=target.parent) as directory:
            temporary=Path(directory)/target.name
            try:
                os.link(cached,temporary)
            except OSError:
                shutil.copyfile(cached,temporary)
            temporary.replace(target)
    stat=target.stat()
    if stat.st_size!=item["bytes"]:
        raise ValueError(f"Unexpected size: {target.name}; original file preserved")
    identity={"size":stat.st_size,"mtime_ns":stat.st_mtime_ns,"sha256":item["sha256"]}
    stamp=verified/(item["sha256"]+".json")
    if not args.rehash and stamp.exists() and json.loads(stamp.read_text())==identity:
        print(f"Verified cache: {target.name}",flush=True)
        continue
    started=time.monotonic()
    with target.open("rb") as stream:
        digest=hashlib.file_digest(stream,"sha256").hexdigest()
    if digest!=item["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {target.name}; original file preserved")
    stamp.write_text(json.dumps(identity)+"\n")
    print(f"SHA-256 verified: {target.name} ({time.monotonic()-started:.1f}s)",flush=True)
