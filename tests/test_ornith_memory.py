import unittest

from paiton_vllm_plugin.runtime.core.utils.ornith_memory import (
    estimate_ornith_memory,
)
from tests.test_ornith_mxfp4_loader import manifest


class OrnithMemoryTests(unittest.TestCase):
    def test_exact_accounting_and_fit(self):
        value = manifest()
        value["memory_planning"] = {
            "activation_blob_bytes": 1000,
            "compiler_owned_constant_bytes": 128,
            "shared_workspace_bytes": 2000,
            "unique_workspace_bytes": 3000,
        }
        estimate = estimate_ornith_memory(
            value,
            available_bytes=64 * 1024**3,
            hybrid_cache_reservation_bytes=1024**3,
        )
        self.assertTrue(estimate.fits)
        self.assertEqual(estimate.hybrid_cache_bytes, 1024**3)
        self.assertEqual(estimate.activation_blob_bytes, 1000)
        self.assertEqual(estimate.workspace_bytes, 5000)
        self.assertGreater(estimate.compiled_unbound_constants_bytes, 0)
        self.assertEqual(estimate.allocator_headroom_bytes, 2 * 1024**3)

    def test_rejects_missing_memory_plan(self):
        with self.assertRaisesRegex(ValueError, "memory-planning"):
            estimate_ornith_memory(manifest())


if __name__ == "__main__":
    unittest.main()
