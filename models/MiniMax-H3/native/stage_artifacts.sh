#!/usr/bin/env bash
# Maintainer step: copy the allowlisted native runtime files (the runner binary and the libraries it loads) from a
# runtime build into artifacts/ and record where they came from. No compiler source is copied.
# Usage: stage_artifacts.sh BUILD_DIR COMPILER_WORKTREE
#   BUILD_DIR: output of native/minimax_h3_video/build_runner.sh; COMPILER_WORKTREE: the checkout it was built from.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build="${1:?BUILD_DIR}"; wt="${2:?COMPILER_WORKTREE}"
out="$here/artifacts"
rm -rf "$out"; mkdir -p "$out/lib"
libs=(h3v_run minimax_h3_image_gemm_gfx1201.so minimax_h3_image_rows_gfx1201.so h3i_qk_prologue_gfx1201.so
      h3i_attention_full8_gfx1201.so h3i_attention_qk8_gfx1201.so h3i_attention_bf16_gfx1201.so
      minimax_h3_image_te_gfx1201.so minimax_h3_video_vae_gfx1201.so minimax_h3_video_enc_audio_gfx1201.so)
for f in "${libs[@]}"; do install -m 0755 "$build/$f" "$out/lib/$f"; done
python3 - "$out" "$wt" "$build" <<'EOF'
import hashlib, json, subprocess, sys
from pathlib import Path
out, wt, build = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
def git(*a): return subprocess.run(["git", "-C", str(wt), *a], check=True, capture_output=True, text=True).stdout.strip()
runtime = dict(commit=git("rev-parse", "HEAD"), dirty_files=len(git("status", "--porcelain", "--", "native").splitlines()))
files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
manifest = dict(components=dict(runtime=runtime), build_components_txt=(build / "COMPONENTS.txt").read_text().splitlines(),
                files=files)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(runtime), "DIRTY" if runtime["dirty_files"] else "clean")
EOF
