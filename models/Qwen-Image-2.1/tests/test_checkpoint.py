"""CPU-only verification and download-boundary checks."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from paiton_image21 import checkpoint


class Checkpoint(unittest.TestCase):
    def fixture(self, root):
        data = b"published model bytes"
        (root / "weight.bin").write_bytes(data)
        return dict(repository="owner/model", revision="a"*40, download_bytes=len(data),
                    files=[dict(file="weight.bin", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())])

    def test_existing_files_verified_offline_without_hub(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); lock=self.fixture(root)
            with patch.object(checkpoint,"checkpoint_lock",return_value=lock), patch.dict(sys.modules,{"huggingface_hub":None}):
                self.assertEqual(checkpoint.prepare(root,offline=True),root)

    def test_tampered_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); lock=self.fixture(root)
            (root/"weight.bin").write_bytes(b"x" * lock["files"][0]["bytes"])
            with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                checkpoint.verify(root,lock)

    def test_download_is_pinned_allowlisted_and_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); lock=self.fixture(root)
            download=Mock(return_value=str(root))
            with patch.object(checkpoint,"checkpoint_lock",return_value=lock), patch.dict(sys.modules,{"huggingface_hub":Mock(snapshot_download=download)}):
                self.assertEqual(checkpoint.prepare(cache_dir="cache",offline=True),root)
            download.assert_called_once_with(repo_id="owner/model",revision="a"*40,
                allow_patterns=["weight.bin"],cache_dir="cache",local_files_only=True,max_workers=2)

    def test_manifest_cannot_escape_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); lock=self.fixture(root)
            lock["files"][0]["file"]="../weight.bin"
            with self.assertRaisesRegex(ValueError,"Invalid checkpoint manifest path"):
                checkpoint.verify(root,lock)


if __name__ == "__main__":
    unittest.main()
