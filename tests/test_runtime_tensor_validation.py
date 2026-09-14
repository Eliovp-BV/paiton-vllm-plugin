from types import SimpleNamespace
import unittest

from paiton_vllm_plugin.runtime.core.model import (
    _check_tensors_contiguous_and_on_gpu,
)
from paiton_vllm_plugin.runtime.core.dtype import (
    dtype_str_to_enum,
    dtype_to_enumerator,
    get_dtype_size,
)


class RuntimeTensorValidationTests(unittest.TestCase):
    def test_uint8_runtime_abi_mapping(self) -> None:
        self.assertEqual(get_dtype_size("uint8"), 1)
        self.assertEqual(dtype_str_to_enum("uint8"), 9)
        self.assertEqual(dtype_to_enumerator("uint8"), "PaitonDtype::kUInt8")

    def test_only_named_exception_may_be_noncontiguous(self) -> None:
        strided = SimpleNamespace(is_cuda=True, is_contiguous=lambda: False)
        contiguous = SimpleNamespace(is_cuda=True, is_contiguous=lambda: True)
        tensors = {"state": strided, "activation": contiguous}

        with self.assertRaisesRegex(ValueError, "'state'.*contiguous"):
            _check_tensors_contiguous_and_on_gpu(tensors, "inputs")
        _check_tensors_contiguous_and_on_gpu(
            tensors, "inputs", noncontiguous_names=frozenset({"state"})
        )

    def test_exception_does_not_allow_host_tensor(self) -> None:
        host = SimpleNamespace(is_cuda=False, is_contiguous=lambda: False)
        with self.assertRaisesRegex(ValueError, "'state'.*on GPU"):
            _check_tensors_contiguous_and_on_gpu(
                {"state": host},
                "inputs",
                noncontiguous_names=frozenset({"state"}),
            )


if __name__ == "__main__":
    unittest.main()
