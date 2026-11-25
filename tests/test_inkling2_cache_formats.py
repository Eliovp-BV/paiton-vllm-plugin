"""The three cache formats of the full-attention visits: what the binding hands the attention library for a vLLM
cache tensor (bf16 strides; fp8 and format 4 pages inside a vLLM block, padded or dense)."""
import pytest

torch = pytest.importorskip("torch")

from paiton_vllm_plugin.inkling2 import cache_plan  # noqa: E402
from paiton_vllm_plugin.inkling2.amd import native_attention as na  # noqa: E402

BLOCKS = 5
# name: (slots per vLLM block, bytes between vLLM blocks): fp8 2 pages in a window-sized block; format 4 three pages
# in a block padded to the window page (today), and four pages in a block of their own size (dense)
PACKED = {"fp8": [(32, 65536)], "kv4": [(48, 65536), (64, 67584)]}


def packed_cache(size, slots, stride):
    return torch.empty(BLOCKS * stride, dtype=torch.uint8).as_strided((BLOCKS, 1, slots, size), (stride, slots * size, size, 1))


def test_formats_agree_with_the_cache_plan():
    assert set(na.KV_FORMATS) | {"bf16"} == set(cache_plan.FULL_BYTES)
    for name, fmt in na.KV_FORMATS.items():
        assert fmt[0] == cache_plan.FULL_BYTES[name]
        assert 0 < fmt[3] < fmt[0] * na.PAGE // 8  # V starts inside a head's part of the page
    assert {fmt[1] for fmt in na.KV_FORMATS.values()} == {512, 2048} and na.KV4_FLAG == na.KV_FORMATS["kv4"][1]


def test_bf16_cache_keeps_vllm_strides():
    cache = na.NativeAttention._cache(torch.empty((BLOCKS, 8, 16, 256), dtype=torch.bfloat16))
    assert (cache.v - cache.k, cache.stride_block, cache.stride_head, cache.stride_slot, cache.block_size) == (
        256, 65536, 8192, 512, 16)


@pytest.mark.parametrize("name,slots,stride", [(n, *layout) for n, layouts in PACKED.items() for layout in layouts])
def test_packed_pages_stay_inside_their_block(monkeypatch, name, slots, stride):
    monkeypatch.setattr(na, "_per_forward", lambda key, make: make())
    fmt = na.KV_FORMATS[name]
    size, page = fmt[0], na.PAGE * fmt[0]
    kv_cache = packed_cache(size, slots, stride)
    cache = na.NativeAttention._cache(kv_cache, fmt)
    assert (cache.v - cache.k, cache.stride_head, cache.stride_slot, cache.block_size) == (fmt[3], page // 8, fmt[4], 16)
    assert stride % cache.stride_block == 0 and page % cache.stride_block == 0

    # writes: vLLM slot -> the library's slot; the library's page starts at (slot // 16) * stride_block
    slot_mapping = torch.arange(BLOCKS * slots, dtype=torch.int64)
    slot_mapping[3] = -1
    mapped = na.NativeAttention._slots(slot_mapping, kv_cache, size)
    starts = set()
    for vllm_slot, lib_slot in zip(slot_mapping.tolist(), mapped.tolist()):
        if vllm_slot < 0:
            assert lib_slot == vllm_slot
            continue
        block, token = divmod(vllm_slot, slots)
        start = (lib_slot // 16) * cache.stride_block
        assert start == block * stride + (token // 16) * page and lib_slot % 16 == token % 16
        assert start + page <= block * stride + stride  # the page ends inside its vLLM block
        starts.add(start)
    assert len(starts) == BLOCKS * slots // 16  # no two pages share bytes

    # reads: a request's blocks -> its pages in token order
    block_table = torch.tensor([[4, 0, 2], [1, 3, 0]], dtype=torch.int32)
    pages = na.NativeAttention._page_table(block_table, kv_cache, size)
    assert pages.shape == (2, 3 * slots // 16)
    for row, blocks in zip(pages.tolist(), block_table.tolist()):
        assert [p * cache.stride_block for p in row] == [b * stride + j * page for b in blocks for j in range(slots // 16)]
