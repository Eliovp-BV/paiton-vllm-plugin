from paiton_vllm_plugin.inkling2.cache_plan import cache_bytes, fixed_bytes, max_context_tokens, plan

GIB = 1 << 30


def test_matches_vllm_refusal_numbers():
    # vLLM 0.32: "32.49 GiB KV cache is needed" for 1,000,000 tokens, max_num_batched_tokens 1024 (2,048 in flight)
    assert round(cache_bytes(1_000_000, 1, 1024) / GIB, 2) == 32.49
    assert round(fixed_bytes(1024) / GIB, 2) == 1.98
    assert round(fixed_bytes(256) / 1e9, 2) == 0.66


def test_matches_vllm_autofit_with_drafter():
    # image c6: "Available KV cache memory: 8.3 GiB", "GPU KV cache size: 206,637 tokens", 12 groups
    assert abs(max_context_tokens(8.3 * GIB, 1, 512, drafter=True) - 206_637) < 64
    assert plan(drafter=True)[2] == 12


def test_packed_formats():
    # a lone bf16 full draft layer next to packed target visits: one group per layer
    assert plan("kv4", True, None)[2] == 99
    # image c6 with --kv-cache fp8 (job p2-kv2): "GPU KV cache size: 469,595 tokens", 13 groups
    assert abs(max_context_tokens(8.3 * GIB, 1, 512, kv="fp8", drafter=True, draft_window=1024) - 469_595) < 64
    assert plan("fp8", True, 1024)[2] == 13
    # format 4: 4 pages of 16,896 bytes per block of 64 tokens (vLLM pads the window and conv pages to 67,584)
    per_token, _, groups = plan("kv4", True, 1024)
    assert (round(per_token), groups) == (8448, 13)


def test_max_context_is_the_inverse():
    for budget, requests, batched in ((10 * GIB, 1, 256), (11 * GIB, 4, 256), (2 * GIB, 1, 1024)):
        tokens = max_context_tokens(budget, requests, batched)
        assert tokens % 16 == 0
        assert cache_bytes(tokens, requests, batched) <= budget < cache_bytes(tokens + 16, requests, batched) or tokens == 0
