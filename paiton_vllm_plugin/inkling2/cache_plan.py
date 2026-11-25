# SPDX-License-Identifier: Apache-2.0
"""Cache budget of Inkling 2 Mini under vLLM 0.32, for the launcher.

Only the 8 full-attention visits (layers 5, 11, 17, 23; 11 and 17 run three
times) keep keys and values for the whole context. The 40 window visits stop at
1,024 tokens and the 48 conv states at 4 tokens, plus the tokens in flight.
The drafter adds two window layers and one full layer (whole context, or a
window with PAITON_INK2_DRAFT_WINDOW). A DFlash2 draft (`drafter="dflash2"`)
adds a bucket of its own: five layers with a 2,048-token window and three
unused caches of the same spec, eight like the full visits, so that vLLM pads
no group. vLLM's own "GPU KV cache size" line is
concurrency times max_model_len and must not be read as a cost per token.

This follows vLLM 0.32's hybrid allocator (`kv_cache_utils.
_get_kv_cache_groups_uniform_page_size`): layers of one spec form a bucket;
pages are raised to the largest page (packed full visits get larger blocks);
every bucket is cut into groups of the smallest bucket's layer count (padding
the last group); one pool block backs one page of every layer of a group. The
window admission rule is `SlidingWindowSpec.max_admission_blocks_per_request`.
Checked against vLLM's own numbers: 32.49 GiB for 1,000,000 tokens at 2,048
tokens in flight (a refused start without drafter), 206,637 tokens in 8.3 GiB
with the drafter (image c6).
"""

FULL_VISITS, WINDOW_VISITS, CONV_STATES = 8, 40, 48
KV_BYTES = 2 * 8 * 128 * 2  # K and V, 8 heads x 128, bf16: one visit, one token
CONV_BYTES = 8 * 1024 * 2  # conv state of one visit and token (832 values padded to 1,024 per head group)
# Bytes per token of one full visit by cache format (native_attention.KV_FORMATS; kv4 = notes/KV-LAYOUT.md format 4).
FULL_BYTES = {"bf16": KV_BYTES, "fp8": 2048, "kv4": 1056}
PER_CONTEXT_TOKEN = FULL_VISITS * KV_BYTES  # 32,768 (bf16, no drafter)
WINDOW, CONV_WINDOW, BLOCK = 1024, 4, 16


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _buckets(kv: str, drafter: bool, draft_window: int | None):
    """(layers, page bytes at the spec's own block size, block size, window or None) per bucket."""
    full = [FULL_VISITS, FULL_BYTES[kv] * BLOCK, BLOCK, None]
    window = [WINDOW_VISITS, KV_BYTES * BLOCK, BLOCK, WINDOW]
    out = [full, window, [CONV_STATES, CONV_BYTES * CONV_WINDOW, CONV_WINDOW, CONV_WINDOW]]
    if drafter == "dflash2":
        out.append([FULL_VISITS, KV_BYTES * BLOCK, BLOCK, 2048])
    elif drafter:
        window[0] += 2  # the draft's local layers 0 and 2
        if draft_window == WINDOW:
            window[0] += 1
        elif kv == "bf16" and draft_window is None:
            full[0] += 1  # same spec as the target's full visits
        else:
            out.append([1, KV_BYTES * BLOCK, BLOCK, draft_window])
    return out


def plan(kv: str = "bf16", drafter: bool | str = False, draft_window: int | None = None, batched_tokens: int = 512,
         concurrent_batches: int = 2) -> tuple[float, int, int, int]:
    """(bytes per context token, fixed bytes per running request, number of cache groups, bytes of one pool block).
    `concurrent_batches` is 2 with vLLM's async scheduling. vLLM 0.32 holds one pool block back (the block pool's null
    block) from both its capacity check and its auto-fit."""
    buckets = _buckets(kv, drafter, draft_window)
    # The plugin declares whole 16-slot pages of the full visits that cover the largest page (format 4: 4 pages,
    # 67,584 bytes); vLLM pads the smaller pages (window and conv, 65,536) to it.
    page = _cdiv(max(b[1] for b in buckets), buckets[0][1]) * buckets[0][1]
    sizes = [b[0] for b in buckets]
    group = max(sizes) if max(sizes) < 1.5 * min(sizes) else min(sizes)
    flight = concurrent_batches * batched_tokens
    per_token, fixed_blocks, groups = 0.0, 0, 0
    for layers, own_page, block, window in buckets:
        block = block * (page // own_page)
        n = _cdiv(layers, group)
        groups += n
        if window is None:
            per_token += n * group * page / block
        else:
            fixed_blocks += n * (_cdiv(window - 1 + flight, block) + 1)
    return per_token, fixed_blocks * group * page, groups, group * page


def fixed_bytes(batched_tokens: int, concurrent_batches: int = 2, **kw) -> int:
    """Bytes one running request needs whatever its context: window caches and conv states, including the tokens
    in flight."""
    return plan(batched_tokens=batched_tokens, concurrent_batches=concurrent_batches, **kw)[1]


def cache_bytes(context_tokens: int, requests: int, batched_tokens: int, **kw) -> int:
    """Cache bytes for `context_tokens` in total over `requests` running requests."""
    per_token, fixed, _, null = plan(batched_tokens=batched_tokens, **kw)
    return int(per_token * context_tokens) + requests * fixed + null


def max_context_tokens(budget_bytes: int, requests: int, batched_tokens: int, **kw) -> int:
    """Context tokens (all requests together, whole blocks of 16) that fit `budget_bytes`."""
    per_token, fixed, _, null = plan(batched_tokens=batched_tokens, **kw)
    return max(0, int((budget_bytes - requests * fixed - null) // per_token) // 16 * 16)
