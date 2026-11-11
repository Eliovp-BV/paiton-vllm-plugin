#!/usr/bin/env bash
# Maintainer step: copy the allowlisted native runtime files from a component build into artifacts/ and record where
# each came from. Usage: H3I_WORKTREES=<dir with wt-gemm wt-attn wt-vae wt-te wt-runtime> stage_artifacts.sh BUILD_DIR
# BUILD_DIR is the output of the runtime's build.sh. No compiler source is copied: only the libraries, the runner
# binary and the four offline weight converters (numpy scripts the container runs on first start).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build="${1:?BUILD_DIR}"; wt="${H3I_WORKTREES:?}"
out="$here/artifacts"
rm -rf "$out"; mkdir -p "$out/lib" "$out/convert"
libs=(h3i_run minimax_h3_image_gemm_gfx1201.so minimax_h3_image_rows_gfx1201.so h3i_qk_prologue_gfx1201.so
      h3i_attention_full8_gfx1201.so h3i_attention_qk8_gfx1201.so h3i_attention_bf16_gfx1201.so
      minimax_h3_image_te_gfx1201.so minimax_h3_image_vae_gfx1201.so)
for f in "${libs[@]}"; do install -m 0755 "$build/$f" "$out/lib/$f"; done
install -m 0644 "$wt/wt-runtime/native/minimax_h3_image/convert.py" "$out/convert/dit_convert.py"
install -m 0644 "$wt/wt-te/native/minimax_h3_image_te/convert.py" "$out/convert/te_convert.py"
install -m 0644 "$wt/wt-gemm/native/minimax_h3_image/real_weights.py" "$out/convert/real_weights.py"
install -m 0644 "$wt/wt-vae/native/minimax_h3_image/pack_weights.py" "$out/convert/vae_pack.py"
python3 - "$out" "$wt" "$build" <<'EOF'
import hashlib, json, subprocess, sys
from pathlib import Path
out, wt, build = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
def git(d, *a): return subprocess.run(["git", "-C", str(d), *a], check=True, capture_output=True, text=True).stdout.strip()
components = {}
for name, sub in [("gemm", "wt-gemm/native/minimax_h3_image"), ("attention", "wt-attn/native/minimax_h3_image"),
                  ("vae", "wt-vae/native/minimax_h3_image"), ("text_encoder", "wt-te/native/minimax_h3_image_te"),
                  ("runtime", "wt-runtime/native/minimax_h3_image")]:
    d = wt / sub
    components[name] = dict(branch=git(d, "rev-parse", "--abbrev-ref", "HEAD"), commit=git(d, "rev-parse", "HEAD"),
                            dirty_files=len(git(d, "status", "--porcelain", "--", ".").splitlines()))
files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
manifest = dict(components=components, build_components_txt=(build / "COMPONENTS.txt").read_text().splitlines(),
                files=files, converters={k: v for k, v in files.items() if k.startswith("convert/")})
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
dirty = [k for k, v in components.items() if v["dirty_files"]]
print(json.dumps(components, indent=1)); print("DIRTY:", dirty or "none")
EOF
