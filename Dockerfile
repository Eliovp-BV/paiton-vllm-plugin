# syntax=docker/dockerfile:1.7

# Assemble the pinned runtime payload in a clean Ubuntu 24.04 filesystem.
# Compiler sources, build caches, SDK development files, and non-gfx1201
# kernel libraries do not cross the stage boundary.
ARG BASE_IMAGE=paiton-vllm@sha256:0101bf315acf1ca1022a4f957e287f9c2105a90a7cab36aad9aa4806c69de5a2
FROM ${BASE_IMAGE} AS qualified

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md /opt/paiton/release-source/
COPY LICENSES /opt/paiton/release-source/LICENSES
COPY paiton_vllm_plugin /opt/paiton/release-source/paiton_vllm_plugin

RUN /opt/venv/bin/python3 -m pip install \
      --no-build-isolation \
      --no-deps \
      --target /opt/paiton/release-plugin \
      /opt/paiton/release-source

# The qualified base keeps some small transitive vLLM dependencies only in its
# development venv. Copy the complete recursive closure of the known missing
# roots, using installed distribution metadata, instead of discovering modules
# one at a time or carrying the 29 GB venv tree.
RUN /opt/venv/bin/python3 - <<'PY'
from collections import deque
from importlib.metadata import distribution
from packaging.markers import default_environment
from packaging.requirements import Requirement
from pathlib import Path
import shutil

roots = {
    "attrs", "boto3", "certifi", "colorama", "iniconfig", "jmespath",
    "markupsafe", "pygments", "python-dateutil", "pyyaml", "rich", "six",
    "urllib3",
}
source_root = Path("/opt/venv/lib/python3.12/site-packages")
output_root = Path("/opt/paiton/runtime-deps")
queue = deque(sorted(roots))
seen: set[str] = set()

while queue:
    name = queue.popleft().lower()
    if name in seen:
        continue
    seen.add(name)
    dist = distribution(name)
    for raw_requirement in dist.requires or ():
        requirement = Requirement(raw_requirement)
        if requirement.marker and not requirement.marker.evaluate(
            default_environment()
        ):
            continue
        queue.append(requirement.name)
    for relative in dist.files or ():
        source = Path(dist.locate_file(relative))
        try:
            destination_relative = source.relative_to(source_root)
        except ValueError:
            continue
        if not source.is_file():
            continue
        destination = output_root / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
PY

# The wheel-installed ROCm SDK trees duplicate /opt/rocm/core-7.14. Import and
# device probes verify that the pinned core tree is authoritative. Retain only
# gfx1201 architecture payloads and remove static/build-only libraries.
RUN set -eux; \
    rm -rf \
      /opt/paiton/site-packages/_rocm_sdk_core \
      /opt/paiton/site-packages/_rocm_sdk_libraries \
      /opt/paiton/venv-site-packages/__editable__.paiton_vllm_plugin-0.1.0.pth \
      /opt/paiton/venv-site-packages/__editable___paiton_vllm_plugin_0_1_0_finder.py \
      /opt/paiton/venv-site-packages/paiton_vllm_plugin-0.1.0.dist-info \
      /opt/rocm/core-7.14/share/miopen; \
    mkdir -p \
      /opt/paiton/site-packages/_rocm_sdk_core \
      /opt/paiton/site-packages/_rocm_sdk_libraries; \
    touch \
      /opt/paiton/site-packages/_rocm_sdk_core/__init__.py \
      /opt/paiton/site-packages/_rocm_sdk_libraries/__init__.py; \
    ln -s /opt/rocm/core-7.14/lib \
      /opt/paiton/site-packages/_rocm_sdk_core/lib; \
    ln -s /opt/rocm/core-7.14/lib \
      /opt/paiton/site-packages/_rocm_sdk_libraries/lib; \
    find /opt/rocm/core-7.14 -type f -name '*.a' -delete; \
    find /opt/rocm/core-7.14/.kpack -type f ! -name '*gfx1201*' -delete; \
    find /opt/rocm/core-7.14/lib -maxdepth 1 \
      -type f -name 'libMIOpenCKGroupedConv_gfx*.so' \
      ! -name '*gfx1201.so' -delete; \
    find /opt/rocm/core-7.14/lib/hipblaslt/library \
      -mindepth 1 -maxdepth 1 -type d ! -name gfx1201 \
      -exec rm -rf '{}' +; \
    find /opt/rocm/core-7.14/lib/rocblas/library \
      -mindepth 1 -maxdepth 1 -type d -name 'gfx*' ! -name gfx1201 \
      -exec rm -rf '{}' +; \
    find /opt/rocm/core-7.14/lib/hipsparselt/library \
      -mindepth 1 -maxdepth 1 -type d -name 'gfx*' ! -name gfx1201 \
      -exec rm -rf '{}' +; \
    find /opt/paiton /opt/rocm/core-7.14 \
      -type d -name __pycache__ -prune -exec rm -rf '{}' +

FROM ubuntu:24.04@sha256:1e0a86e57d247923571b75e0aaf48a1449cf8c543d51fb3e07a4a7d7bfa79316

ARG PLUGIN_REVISION
RUN set -eu; \
    case "${PLUGIN_REVISION}" in *[!0-9a-f]*|'') exit 1;; esac; \
    test "${#PLUGIN_REVISION}" -eq 40
LABEL org.opencontainers.image.title="Paiton Qwen3.8 Qronos RDNA4 server" \
      org.opencontainers.image.description="One-command gfx1201 W4A16 serving runtime" \
      org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.revision="${PLUGIN_REVISION}" \
      org.opencontainers.image.version="paiton-qwen38-qronos-w4a16-gfx1201-v1.3.0" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.base.name="docker.io/library/ubuntu:24.04" \
      org.opencontainers.image.base.digest="sha256:1e0a86e57d247923571b75e0aaf48a1449cf8c543d51fb3e07a4a7d7bfa79316" \
      dev.paiton.qualified-base.digest="sha256:0101bf315acf1ca1022a4f957e287f9c2105a90a7cab36aad9aa4806c69de5a2" \
      dev.paiton.model="amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16" \
      dev.paiton.model.revision="649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2" \
      dev.paiton.artifact.sha256="be534b72dc47094e1d067594e8642174909f7ab223676daa29d0f9a95c5b55b3" \
      dev.paiton.artifact-manifest.sha256="18bd30d71285356472f7482ed7ec45dbba6f733ad902f0059ad3e1364edb2f23" \
      dev.paiton.config.sha256="4bb0ded506a18ccadc84b4c97426df66042309042a23d63109191fdff8434c2e" \
      dev.paiton.lm-head-artifact.sha256="6c5d36808b4785bd0cfb6f7d7f76f54b8e3498b5d9e92eb79a5fba5ecf866ae9" \
      dev.paiton.lm-head-artifact-manifest.sha256="f35ae96417046b7a3c5fc524ae5cbf1f95af93bcc52519dc84690238673d97fa" \
      dev.paiton.release-metadata.sha256="6dddd10ebb5bd58ca292b50fa1190ecfa3895642bd553940c02de6b976e438ac" \
      dev.paiton.gpu.arch="gfx1201" \
      dev.paiton.rocm.version="7.14.60850" \
      dev.paiton.vllm.version="0.28.0.dev0+eliovp.quark48606.g39bd959b5.rocm714" \
      dev.paiton.vllm.revision="39bd959b582c85e78e7e0326d49042ce7c3c07ed"

# Ubuntu supplies the stable host layout. The qualified stage contributes its
# exact Python/host runtime and only the pinned Paiton/ROCm payloads.
COPY --from=qualified /usr /usr
COPY --from=qualified /var/lib/dpkg /var/lib/dpkg
COPY --from=qualified /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=qualified /opt/venv/bin /opt/venv/bin
COPY --from=qualified /opt/venv/pyvenv.cfg /opt/venv/pyvenv.cfg
COPY --from=qualified /opt/rocm/core-7.14 /opt/rocm/core-7.14
COPY --from=qualified /opt/paiton/site-packages /opt/paiton/site-packages
COPY --from=qualified /opt/paiton/venv-site-packages /opt/paiton/venv-site-packages
COPY --from=qualified /opt/paiton/runtime-deps /opt/paiton/venv-site-packages
COPY --from=qualified /opt/paiton/vllm /opt/paiton/vllm
COPY --from=qualified /opt/paiton/release-plugin /opt/paiton/release-plugin
COPY --from=qualified /opt/paiton/release-source/LICENSE /licenses/paiton/LICENSE
COPY --from=qualified /opt/paiton/release-source/LICENSE /licenses/third-party/Apache-2.0.txt
COPY --from=qualified /opt/paiton/release-source/LICENSES /licenses/third-party
COPY --from=qualified /opt/paiton/release-source/THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md

# Record the pinned vLLM source revision with an explicit runtime version.
# Only generated version and installation metadata are changed here.
RUN /opt/venv/bin/python3 - <<'PY'
import base64
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

site_packages = Path("/opt/paiton/venv-site-packages")
source_version = Path("/opt/paiton/vllm/vllm/_version.py")
old_version = "0.1.dev1+g39bd959b5.rocm714"
old_source_version = "0.1.dev1+g39bd959b5"
new_version = "0.28.0.dev0+eliovp.quark48606.g39bd959b5.rocm714"
revision = "39bd959b582c85e78e7e0326d49042ce7c3c07ed"

old_dist_info = site_packages / f"vllm-{old_version}.dist-info"
new_dist_info = site_packages / f"vllm-{new_version}.dist-info"
if not old_dist_info.is_dir() or new_dist_info.exists():
    raise SystemExit("unexpected vLLM distribution metadata layout")

version_text = source_version.read_text(encoding="utf-8")
old_assignment = f"__version__ = version = '{old_source_version}'"
new_assignment = f"__version__ = version = '{new_version}'"
if version_text.count(old_assignment) != 1:
    raise SystemExit("unexpected vLLM generated version file")
version_text = version_text.replace(old_assignment, new_assignment)
version_text = re.sub(
    r"^__version_tuple__ = version_tuple = .*?$",
    "__version_tuple__ = version_tuple = "
    "(0, 28, 0, 'dev0', "
    "'eliovp.quark48606.g39bd959b5.rocm714')",
    version_text,
    count=1,
    flags=re.MULTILINE,
)
if f"__commit_id__ = commit_id = 'g{revision[:9]}'" not in version_text:
    raise SystemExit("unexpected vLLM commit identifier")
source_version.write_text(version_text, encoding="utf-8")

metadata = old_dist_info / "METADATA"
metadata_text = metadata.read_text(encoding="utf-8")
old_metadata_version = f"Version: {old_version}\n"
if metadata_text.count(old_metadata_version) != 1:
    raise SystemExit("unexpected vLLM METADATA version")
metadata.write_text(
    metadata_text.replace(old_metadata_version, f"Version: {new_version}\n"),
    encoding="utf-8",
)
(old_dist_info / "direct_url.json").write_text(
    json.dumps(
        {
            "url": "https://github.com/vllm-project/vllm.git",
            "vcs_info": {
                "vcs": "git",
                "commit_id": revision,
                "requested_revision": revision,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n",
    encoding="utf-8",
)

# PYTHONPATH points directly at /opt/paiton/vllm. Remove only vLLM's copied
# editable-install finder, which refers to the temporary build directory.
editable_vllm_files = (
    site_packages / f"__editable__.vllm-{old_version}.pth",
    site_packages
    / "__editable___vllm_0_1_dev1_g39bd959b5_rocm714_finder.py",
    site_packages
    / "__pycache__/__editable___vllm_0_1_dev1_g39bd959b5_rocm714_finder.cpython-312.pyc",
)
for path in editable_vllm_files:
    if path.exists():
        path.unlink()

old_record = old_dist_info / "RECORD"
with old_record.open(newline="", encoding="utf-8") as source:
    rows = list(csv.reader(source))
# Docker overlay files copied from an earlier layer cannot always be renamed
# across layer boundaries. Materialize the corrected directory in this layer,
# then remove the obsolete name.
shutil.copytree(old_dist_info, new_dist_info)
shutil.rmtree(old_dist_info)

def digest(path: Path) -> tuple[str, str]:
    payload = path.read_bytes()
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}", str(len(payload))

new_rows: list[list[str]] = []
for row in rows:
    relative = row[0]
    if relative.startswith("__editable__") or relative.startswith(
        "__pycache__/__editable__"
    ):
        continue
    if relative == "../../../bin/vllm":
        relative = "../../venv/bin/vllm"
    relative = relative.replace(
        f"vllm-{old_version}.dist-info/",
        f"vllm-{new_version}.dist-info/",
        1,
    )
    if relative.endswith("/RECORD"):
        new_rows.append([relative, "", ""])
        continue
    candidate = site_packages / relative
    if candidate.is_file():
        file_hash, file_size = digest(candidate)
        new_rows.append([relative, file_hash, file_size])
    else:
        new_rows.append([relative, *row[1:]])

new_record = new_dist_info / "RECORD"
with new_record.open("w", newline="", encoding="utf-8") as output:
    csv.writer(output, lineterminator="\n").writerows(new_rows)
PY

COPY models/Qwen3.8/paiton-release.json /opt/paiton/qwen38-overlay/paiton-release.json
COPY --chmod=0555 models/Qwen3.8/paiton-chat /usr/local/bin/paiton-chat
COPY --from=qwen38_overlay /config.json /opt/paiton/qwen38-overlay/config.json
COPY --from=qwen38_overlay /Qwen3.8-27B-Quark-Qronos-INT4-W4A16_gfx1201_tp1_mt8192_ctx8192.so /opt/paiton/qwen38-overlay/Qwen3.8-27B-Quark-Qronos-INT4-W4A16_gfx1201_tp1_mt8192_ctx8192.so
COPY --from=qwen38_overlay /Qwen3.8-27B-Quark-Qronos-INT4-W4A16_gfx1201_tp1_mt8192_ctx8192.manifest.json /opt/paiton/qwen38-overlay/Qwen3.8-27B-Quark-Qronos-INT4-W4A16_gfx1201_tp1_mt8192_ctx8192.manifest.json
COPY --from=qwen38_overlay /benchmark_w4a16_n248320_k5120_gfx1201.so /opt/paiton/qwen38-overlay/benchmark_w4a16_n248320_k5120_gfx1201.so
COPY --from=qwen38_overlay /benchmark_w4a16_n248320_k5120_gfx1201.manifest.json /opt/paiton/qwen38-overlay/benchmark_w4a16_n248320_k5120_gfx1201.manifest.json

RUN set -eu; \
    cd /opt/paiton/qwen38-overlay; \
    test "$(stat -c %s paiton-release.json)" = 2431; \
    test "$(stat -c %s config.json)" = 16298; \
    test "$(stat -c %s Qwen3.8-27B-Quark-Qronos-INT4-W4A16_gfx1201_tp1_mt8192_ctx8192.so)" = 5834720; \
    test "$(stat -c %s Qwen3.8-27B-Quark-Qronos-INT4-W4A16_gfx1201_tp1_mt8192_ctx8192.manifest.json)" = 963895; \
    test "$(stat -c %s benchmark_w4a16_n248320_k5120_gfx1201.so)" = 385736; \
    test "$(stat -c %s benchmark_w4a16_n248320_k5120_gfx1201.manifest.json)" = 2501; \
    printf '%s  %s\n' \
      6dddd10ebb5bd58ca292b50fa1190ecfa3895642bd553940c02de6b976e438ac paiton-release.json \
      4bb0ded506a18ccadc84b4c97426df66042309042a23d63109191fdff8434c2e config.json \
      be534b72dc47094e1d067594e8642174909f7ab223676daa29d0f9a95c5b55b3 Qwen3.8-27B-Quark-Qronos-INT4-W4A16_gfx1201_tp1_mt8192_ctx8192.so \
      18bd30d71285356472f7482ed7ec45dbba6f733ad902f0059ad3e1364edb2f23 Qwen3.8-27B-Quark-Qronos-INT4-W4A16_gfx1201_tp1_mt8192_ctx8192.manifest.json \
      6c5d36808b4785bd0cfb6f7d7f76f54b8e3498b5d9e92eb79a5fba5ecf866ae9 benchmark_w4a16_n248320_k5120_gfx1201.so \
      f35ae96417046b7a3c5fc524ae5cbf1f95af93bcc52519dc84690238673d97fa benchmark_w4a16_n248320_k5120_gfx1201.manifest.json \
      | sha256sum -c -; \
    chmod 0444 ./*

ENV ROCM_PATH=/opt/rocm/core-7.14 \
    HIP_PATH=/opt/rocm/core-7.14 \
    PATH=/opt/rocm/core-7.14/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:/opt/rocm/core-7.14/lib64 \
    PYTHONPATH=/opt/paiton/release-plugin:/opt/paiton/venv-site-packages:/opt/paiton/site-packages:/opt/paiton/vllm \
    VLLM_USE_PAITON_PLATFORM=1 \
    VLLM_DISABLE_PAITON_PLATFORM=0 \
    PAITON_GPU_ARCH=gfx1201 \
    PYTORCH_ROCM_ARCH=gfx1201 \
    AITER_ROCM_ARCH=gfx1201 \
    ROCM_SDK_TARGET_FAMILY=gfx1201 \
    HF_HOME=/models/cache \
    HF_HUB_CACHE=/models/cache/hub \
    PAITON_BASE_HF_CACHE=/models/base-cache \
    PAITON_BUNDLED_OVERLAY=/opt/paiton/qwen38-overlay

# Fail the image build if any direct or transitive runtime requirement was
# omitted by the slim packaging boundary.
RUN python3 - <<'PY'
from collections import deque
from importlib.metadata import PackageNotFoundError, distribution
from packaging.markers import default_environment
from packaging.requirements import Requirement

queue = deque(["vllm", "paiton-vllm-plugin"])
seen: set[str] = set()
missing: dict[str, set[str]] = {}
incompatible: dict[str, set[str]] = {}
while queue:
    name = queue.popleft().lower()
    if name in seen:
        continue
    seen.add(name)
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        missing.setdefault(name, set()).add("runtime root")
        continue
    for raw_requirement in dist.requires or ():
        requirement = Requirement(raw_requirement)
        if requirement.marker and not requirement.marker.evaluate(
            default_environment()
        ):
            continue
        try:
            installed = distribution(requirement.name)
        except PackageNotFoundError:
            missing.setdefault(requirement.name, set()).add(dist.metadata["Name"])
        else:
            if requirement.specifier and not requirement.specifier.contains(
                installed.version, prereleases=True
            ):
                incompatible.setdefault(requirement.name, set()).add(
                    f"{installed.version} does not satisfy {requirement.specifier} "
                    f"for {dist.metadata['Name']}"
                )
            queue.append(requirement.name)
if missing:
    details = ", ".join(
        f"{name} (from {sorted(parents)})"
        for name, parents in sorted(missing.items())
    )
    raise SystemExit(f"missing recursive runtime dependencies: {details}")
if incompatible:
    details = ", ".join(
        f"{name} ({sorted(problems)})"
        for name, problems in sorted(incompatible.items())
    )
    raise SystemExit(f"incompatible recursive runtime dependencies: {details}")
PY

RUN python3 - <<'PY'
import base64
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from runpy import run_path

expected = "0.28.0.dev0+eliovp.quark48606.g39bd959b5.rocm714"
expected_commit = "g39bd959b5"
expected_revision = "39bd959b582c85e78e7e0326d49042ce7c3c07ed"
expected_tuple = (0, 28, 0, "dev0", "eliovp.quark48606.g39bd959b5.rocm714")
module_metadata = run_path("/opt/paiton/vllm/vllm/_version.py")
if version("vllm") != expected or module_metadata["__version__"] != expected:
    raise SystemExit(
        f"incoherent vLLM version: metadata={version('vllm')!r}, "
        f"module={module_metadata['__version__']!r}"
    )
if module_metadata["__commit_id__"] != expected_commit:
    raise SystemExit(
        f"unexpected vLLM commit: {module_metadata['__commit_id__']!r}"
    )
if module_metadata["__version_tuple__"] != expected_tuple:
    raise SystemExit(
        f"unexpected vLLM version tuple: {module_metadata['__version_tuple__']!r}"
    )

site_packages = Path("/opt/paiton/venv-site-packages")
dist_infos = list(site_packages.glob("vllm-*.dist-info"))
if [path.name for path in dist_infos] != [f"vllm-{expected}.dist-info"]:
    raise SystemExit(f"unexpected vLLM dist-info directories: {dist_infos!r}")
dist_info = dist_infos[0]
direct_url = json.loads((dist_info / "direct_url.json").read_text(encoding="utf-8"))
if direct_url != {
    "url": "https://github.com/vllm-project/vllm.git",
    "vcs_info": {
        "vcs": "git",
        "commit_id": expected_revision,
        "requested_revision": expected_revision,
    },
}:
    raise SystemExit(f"unexpected vLLM direct_url.json: {direct_url!r}")
if list(site_packages.glob("__editable__.vllm-*")) or list(
    site_packages.glob("__editable___vllm_*")
):
    raise SystemExit("obsolete vLLM editable-install files remain")

with (dist_info / "RECORD").open(newline="", encoding="utf-8") as source:
    rows = list(csv.reader(source))
record_paths = {row[0] for row in rows}
if "../../venv/bin/vllm" not in record_paths or any(
    "0.1.dev1" in path or "/tmp/vllm-pinned-source" in path
    for path in record_paths
):
    raise SystemExit("vLLM RECORD retains obsolete installation paths")
for relative, recorded_hash, recorded_size in rows:
    if not recorded_hash:
        continue
    candidate = site_packages / relative
    if not candidate.is_file():
        raise SystemExit(f"vLLM RECORD file is missing: {relative}")
    payload = candidate.read_bytes()
    actual_hash = "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(payload).digest()
    ).rstrip(b"=").decode("ascii")
    if recorded_hash != actual_hash or recorded_size != str(len(payload)):
        raise SystemExit(f"vLLM RECORD mismatch: {relative}")
PY

RUN python3 - <<'PY'
from importlib.metadata import distributions
from pathlib import Path

matches = [
    dist
    for dist in distributions()
    if dist.metadata["Name"] == "paiton-vllm-plugin"
]
if len(matches) != 1:
    raise SystemExit(
        "expected exactly one paiton-vllm-plugin distribution, found "
        + ", ".join(f"{dist.version} at {dist._path}" for dist in matches)
    )
plugin = matches[0]
expected_root = Path("/opt/paiton/release-plugin")
if plugin.version != "0.2.0" or plugin._path.parent != expected_root:
    raise SystemExit(
        f"unexpected Paiton plugin distribution: {plugin.version} at {plugin._path}"
    )

PY

WORKDIR /opt/paiton
EXPOSE 8000
VOLUME ["/models/cache"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=30m --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
ENTRYPOINT ["python3", "-m", "paiton_vllm_plugin.qwen38_release_server"]
