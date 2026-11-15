#!/usr/bin/env bash
# Maintainer step: copy the allowlisted native runtime files into artifacts/ and record which commit each was built from.
# Usage: LTX_WORKTREES=<dir with wt-gemm wt-attn wt-vae wt-te wt-runtime> \
#        LTX_COMMITS="gemm=<sha> attention=<sha> vae=<sha> text_encoder=<sha> runtime=<sha>" stage_artifacts.sh RUNNER LIB_DIR...
# RUNNER is the ltx_run binary from the runtime's build.sh; each LIB_DIR holds component libraries (the first copy of a
# file name wins). Build every file from `git archive` of the commit named in LTX_COMMITS; the script checks that each
# commit is on its component's branch. No compiler source is copied, only the libraries and the runner binary. The
# converted weights come from the Hugging Face bundle in checkpoints.lock.json.
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
python3 - "$out" "$wt" "${LTX_COMMITS:?}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
out, wt, commits = Path(sys.argv[1]), Path(sys.argv[2]), dict(c.split("=") for c in sys.argv[3].split())
def git(d, *a): return subprocess.run(["git", "-C", str(d), *a], check=True, capture_output=True, text=True).stdout.strip()
components = {}
for name, sub in [("gemm", "wt-gemm"), ("attention", "wt-attn"), ("vae", "wt-vae"), ("text_encoder", "wt-te"), ("runtime", "wt-runtime")]:
    d = wt / sub
    branch = git(d, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git(d, "rev-parse", "--verify", commits[name] + "^{commit}")
    git(d, "merge-base", "--is-ancestor", commit, branch)   # fails unless the commit is on the component's branch
    components[name] = dict(branch=branch, commit=commit)
files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
(out / "manifest.json").write_text(json.dumps(dict(components=components, files=files), indent=2) + "\n")
print(json.dumps(components, indent=1))
PY
