from paiton_vllm_plugin.inkling2.cache_plan import cache_bytes, fixed_bytes, max_context_tokens

GIB = 1 << 30


def test_matches_vllm_refusal_numbers():
    # vLLM 0.32: "32.49 GiB KV cache is needed" for 1,000,000 tokens, max_num_batched_tokens 1024 (2,048 in flight)
    assert round(cache_bytes(1_000_000, 1, 1024) / GIB, 2) == 32.49
    assert round(fixed_bytes(1024) / GIB, 2) == 1.97
    assert round(fixed_bytes(256) / 1e9, 2) == 0.66


def test_max_context_is_the_inverse():
    for budget, requests, batched in ((10 * GIB, 1, 256), (11 * GIB, 4, 256), (2 * GIB, 1, 1024)):
        tokens = max_context_tokens(budget, requests, batched)
        assert tokens % 16 == 0
        assert cache_bytes(tokens, requests, batched) <= budget < cache_bytes(tokens + 16, requests, batched) or tokens == 0
