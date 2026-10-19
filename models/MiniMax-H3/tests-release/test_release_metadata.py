"""CPU tests for the v1.1.0 release lock, manifest contract and verifier."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

DRAFTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRAFTS))

import verify as release_verify  # noqa: E402

TAG = "ghcr.io/eliovp/paiton-vllm-plugin:minimax-h3-rdna4-v1.1.0"
DIGEST = "sha256:" + "1" * 64
BASE_DIGEST = "sha256:" + "2" * 64
RELEASE_ID = "paiton-minimax-h3-ref2va-rdna4-v1.1.0"
ARTIFACT_NAME = "minimax_h3_projections_gfx1201.so"
NOTICE = (
    "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, "
    "Copyright \u00a9 2026 MiniMax. All Rights Reserved.\n"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_bytes(payload: bytes, **overrides):
    manifest = {
        "manifest_version": 1,
        "binary_abi_version": 1,
        "capability_version": 1,
        "h3_projection_abi_version": 1,
        "h3_contract_version": 1,
        "h3_max_tokens": 49152,
        "h3_packed_layout": ["text", "refs", "audio", "video"],
        "h3_ref_policy": {
            "order": ["image", "video_audio", "audio"],
            "timestep": "max(t,0.999)",
            "audio_timestep": "max(t,1.0)",
            "audio_before_video": True,
            "rope": "float64->float32[S,96]",
        },
        "h3_ref_limits": {"images": 9, "videos": 3, "audios": 3},
        "h3_capabilities": ["fl2va", "ref2va"],
        "target": {"arch": "gfx1201", "wave_size": 32},
        "parallelism": {"tp_size": 1},
        "model": {
            "repository": "Comfy-Org/MiniMax-H3",
            "revision": "a98869194787969724c7425d95d0ed73ce9202af",
            "scope": "ref2va_pruned_int8_accumulation_projections",
        },
        "artifact": {
            "filename": ARTIFACT_NAME,
            "size_bytes": len(payload),
            "sha256": sha256(payload),
        },
    }
    manifest.update(overrides)
    return json.dumps(manifest).encode()


def lock_dict(payload: bytes):
    return {
        "schema_version": 1,
        "release": {"id": RELEASE_ID, "version": "1.1.0", "tag": "minimax-h3-rdna4-v1.1.0"},
        "artifact": {
            "file": f"artifacts/{ARTIFACT_NAME}",
            "manifest": f"artifacts/{ARTIFACT_NAME}.manifest.json",
            "size_bytes": len(payload),
            "sha256": sha256(payload),
        },
        "artifact_allowlist": [
            f"artifacts/{ARTIFACT_NAME}",
            f"artifacts/{ARTIFACT_NAME}.manifest.json",
        ],
        "image": {"tag": TAG, "digest": DIGEST, "base_digest": BASE_DIGEST},
        "contract": {
            "h3_contract_version": 1,
            "h3_max_tokens": 49152,
            "ref_limits": {"images": 9, "videos": 3, "audios": 3},
            "packed_layout": ["text", "refs", "audio", "video"],
        },
        "publication": {
            "github_repository": "Eliovp-BV/paiton-vllm-plugin",
            "huggingface_repository": "EliovpAI/MiniMax-H3-W4A8-Paiton-RDNA4",
            "huggingface_revision": "paiton-rdna4-v1.1.0",
        },
    }


@pytest.fixture()
def release(tmp_path: Path) -> Path:
    payload = b"\x7fELF" + b"paiton-h3" * 64
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / ARTIFACT_NAME).write_bytes(payload)
    (tmp_path / "artifacts" / f"{ARTIFACT_NAME}.manifest.json").write_bytes(
        manifest_bytes(payload)
    )
    (tmp_path / "release.lock.json").write_text(json.dumps(lock_dict(payload)))
    (tmp_path / "LICENSE").write_text("Apache License 2.0\n")
    (tmp_path / "NOTICE").write_text(NOTICE)
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_text(
        "MiniMax H3, comfy-kitchen, ComfyUI, AIMDO\n"
    )
    (tmp_path / "MODIFICATIONS.md").write_text("Modified comfy-kitchen attention.\n")
    (tmp_path / "LICENSES").mkdir()
    for name in (
        "MiniMax-H3-Community.txt",
        "comfy-kitchen-Apache-2.0.txt",
        "ComfyUI-GPL-3.0.txt",
        "Triton-MIT.txt",
    ):
        (tmp_path / "LICENSES" / name).write_text(f"{name}\n")
    (tmp_path / "README.md").write_text(f"# H3\n\n`{TAG}@{DIGEST}`\n")
    return tmp_path


def test_valid_release_passes(release: Path):
    result = release_verify.verify(release, require_artifact=True, require_digest=True)
    assert result["status"] == "passed"
    assert result["artifact_present"] is True
    assert result["artifact_sha256"] == sha256(
        (release / "artifacts" / ARTIFACT_NAME).read_bytes()
    )


def test_artifact_mismatch_fails(release: Path):
    (release / "artifacts" / ARTIFACT_NAME).write_bytes(b"tampered")
    with pytest.raises(release_verify.VerificationError) as error:
        release_verify.verify(release, require_artifact=True)
    assert "size" in str(error.value) or "sha256" in str(error.value)
    assert error.value.code == 2


def test_manifest_requires_ref_limits(release: Path):
    payload = (release / "artifacts" / ARTIFACT_NAME).read_bytes()
    manifest = manifest_bytes(payload)
    data = json.loads(manifest)
    del data["h3_ref_limits"]
    (release / "artifacts" / f"{ARTIFACT_NAME}.manifest.json").write_text(json.dumps(data))
    with pytest.raises(release_verify.VerificationError) as error:
        release_verify.verify(release, require_artifact=True)
    assert "h3_ref_limits" in str(error.value)


def test_manifest_requires_ref2va_capability(release: Path):
    payload = (release / "artifacts" / ARTIFACT_NAME).read_bytes()
    data = json.loads(manifest_bytes(payload))
    data["h3_capabilities"] = ["fl2va"]
    (release / "artifacts" / f"{ARTIFACT_NAME}.manifest.json").write_text(json.dumps(data))
    with pytest.raises(release_verify.VerificationError) as error:
        release_verify.verify(release, require_artifact=True)
    assert "ref2va" in str(error.value)


def test_notice_sentence_required(release: Path):
    (release / "NOTICE").write_text("Paiton MiniMax H3 runtime\n")
    with pytest.raises(release_verify.VerificationError) as error:
        release_verify.verify(release)
    assert "MiniMax H3" in str(error.value)


def test_notice_sentence_may_be_wrapped(release: Path):
    (release / "NOTICE").write_text(
        "Paiton MiniMax H3 runtime\n"
        "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement,\n"
        "Copyright \u00a9 2026 MiniMax. All Rights Reserved.\n"
    )
    result = release_verify.verify(release)
    assert result["status"] == "passed"


def test_forbidden_private_path(release: Path):
    (release / "MODIFICATIONS.md").write_text("built in /home/testbench/paiton-compiler\n")
    with pytest.raises(release_verify.VerificationError) as error:
        release_verify.verify(release)
    assert "forbidden" in str(error.value)


def test_readme_digest_is_advisory(release: Path):
    (release / "README.md").write_text("# H3\n")
    result = release_verify.verify(release, require_digest=True)
    assert result["status"] == "passed"


def test_sha256sums_checked(release: Path):
    entries = []
    for relative in ("LICENSE", "NOTICE", "README.md"):
        digest = sha256((release / relative).read_bytes())
        entries.append(f"{digest}  {relative}")
    (release / "SHA256SUMS").write_text("\n".join(entries) + "\n")
    result = release_verify.verify(release)
    assert result["checksums_checked"] == 3


def test_missing_artifact_allowed_in_metadata_mode(release: Path):
    (release / "artifacts" / ARTIFACT_NAME).unlink()
    result = release_verify.verify(release)
    assert result["artifact_present"] is False


def test_floating_tag_rejected_in_lock(release: Path):
    lock = json.loads((release / "release.lock.json").read_text())
    lock["image"]["digest"] = "latest"
    (release / "release.lock.json").write_text(json.dumps(lock))
    with pytest.raises(release_verify.VerificationError) as error:
        release_verify.verify(release)
    assert "digest" in str(error.value)


def test_unlisted_so_rejected(release: Path):
    (release / "artifacts" / "extra_generated.so").write_bytes(b"smuggled")
    with pytest.raises(release_verify.VerificationError) as error:
        release_verify.verify(release)
    assert "unlisted runtime artifact" in str(error.value)


def test_allowlist_must_include_artifact(release: Path):
    lock = json.loads((release / "release.lock.json").read_text())
    lock["artifact_allowlist"] = [f"artifacts/{ARTIFACT_NAME}.manifest.json"]
    (release / "release.lock.json").write_text(json.dumps(lock))
    with pytest.raises(release_verify.VerificationError) as error:
        release_verify.verify(release)
    assert "does not include" in str(error.value)


def test_allowlist_rejects_parent_paths(release: Path):
    lock = json.loads((release / "release.lock.json").read_text())
    lock["artifact_allowlist"].append("../escape.so")
    (release / "release.lock.json").write_text(json.dumps(lock))
    with pytest.raises(release_verify.VerificationError) as error:
        release_verify.verify(release)
    assert "unsafe" in str(error.value)

