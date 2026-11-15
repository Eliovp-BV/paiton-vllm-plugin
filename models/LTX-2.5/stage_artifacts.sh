#!/usr/bin/env bash
# Maintainer step: copy the allowlisted native runtime files into artifacts/ and record where each came from.
# Usage: LTX_WORKTREES=<dir with wt-gemm wt-attn wt-vae wt-te wt-runtime> stage_artifacts.sh RUNNER LIB_DIR...
# RUNNER is the ltx_run binary from the runtime's build.sh; each LIB_DIR holds component libraries (the first copy of a
# file name wins). No compiler source is copied, only the libraries and the runner binary. The converted weights come
# from the Hugging Face bundle in checkpoints.lock.json. The manifest records each worktree HEAD at staging time: a
# release build rebuilds every library from those committed HEADs (no dirty files) before staging.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="${1:?RUNNER}"; shift; wt="${LTX_WORKTREES:?}"
out="$here/artifacts"
rm -rf "$out"; mkdir -p "$out/lib"
install -m 0755 "$runner" "$out/lib/ltx_run"
for f in ltx25_gemm_gfx1201.so ltx25_rows_gfx1201.so ltx25_attention_gfx1201.so ltx25_te_gfx1201.so \
         ltx25_vae_gfx1201.so ltx25_audio_gfx1201.so; do
  src="$(find "$@" -name "$f" -type f -print -quit)"
  [ -n "$src" ] || { echo "missing $f in $*" >&2; exit 1; }
  install -m 0755 "$src" "$out/lib/$f"; echo "$f <- $src"
done
python3 - "$out" "$wt" <<'EOF'
import hashlib, json, subprocess, sys
from pathlib import Path
out, wt = Path(sys.argv[1]), Path(sys.argv[2])
def git(d, *a): return subprocess.run(["git", "-C", str(d), *a], check=True, capture_output=True, text=True).stdout.strip()
components = {}
for name, sub in [("gemm", "wt-gemm"), ("attention", "wt-attn"), ("vae", "wt-vae"), ("text_encoder", "wt-te"), ("runtime", "wt-runtime")]:
    d = wt / sub / "native/ltx25"
    components[name] = dict(branch=git(d, "rev-parse", "--abbrev-ref", "HEAD"), worktree_head=git(d, "rev-parse", "HEAD"),
                            dirty_files=len(git(d, "status", "--porcelain", "--", ".").splitlines()))
files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
(out / "manifest.json").write_text(json.dumps(dict(components=components, files=files), indent=2) + "\n")
dirty = [k for k, v in components.items() if v["dirty_files"]]
print(json.dumps(components, indent=1)); print("DIRTY:", dirty or "none")
EOF
