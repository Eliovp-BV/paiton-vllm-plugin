# SPDX-License-Identifier: Apache-2.0
"""Cache budget of Inkling 2 Mini under vLLM 0.32, for the launcher.

Only the 8 full-attention visits (layers 5, 11, 17, 23; 11 and 17 run three
times) keep keys and values for the whole context. The 40 window visits stop at
1,024 tokens and the 48 conv states at 4 tokens, plus the tokens in flight.
vLLM's own "GPU KV cache size" line is concurrency times max_model_len and
must not be read as a cost per token.

Formula from vLLM 0.32's admission rule (`SlidingWindowSpec.
max_admission_blocks_per_request`), confirmed by one start that vLLM refused
with its own numbers: 32.49 GiB for 1,000,000 tokens at 2,048 tokens in flight.
"""

FULL_VISITS, WINDOW_VISITS, CONV_STATES = 8, 40, 48
KV_BYTES = 2 * 8 * 128 * 2  # K and V, 8 heads x 128, bf16: one visit, one token
CONV_BYTES = 8 * 1024 * 2  # conv state of one visit and token (832 values padded to 1,024 per head group)
PER_CONTEXT_TOKEN = FULL_VISITS * KV_BYTES  # 32,768


def fixed_bytes(batched_tokens: int, concurrent_batches: int = 2) -> int:
    """Bytes one running request needs whatever its context: window caches
    and conv states, including the tokens in flight (`concurrent_batches` is 2
    with vLLM's async scheduling)."""
    flight = concurrent_batches * batched_tokens
    return WINDOW_VISITS * (1023 + flight) * KV_BYTES + CONV_STATES * (3 + flight) * CONV_BYTES


def cache_bytes(context_tokens: int, requests: int, batched_tokens: int) -> int:
    """Cache bytes for `context_tokens` in total over `requests` running requests."""
    return PER_CONTEXT_TOKEN * context_tokens + requests * fixed_bytes(batched_tokens)


def max_context_tokens(budget_bytes: int, requests: int, batched_tokens: int) -> int:
    """Context tokens (all requests together, whole blocks of 16) that fit `budget_bytes`."""
    free = budget_bytes - requests * fixed_bytes(batched_tokens)
    return max(0, free // PER_CONTEXT_TOKEN // 16 * 16)
