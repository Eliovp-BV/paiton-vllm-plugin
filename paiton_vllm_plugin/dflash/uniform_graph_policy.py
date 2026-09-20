"""Two exact uniform target shapes; no draft choices or tensor computation."""
CASES = ((8, 4), (8, 5))
MAX_EXTRA_BYTES = 64 * 1024 * 1024


def eligible(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len):
    return (
        (num_reqs, uniform_token_count) in CASES
        and num_tokens == num_reqs * uniform_token_count
        and max_query_len == uniform_token_count
        and num_active_loras == 0
    )


def extend_descriptors(original, factory, full_mode):
    """Keep existing descriptors; add exact cases only to the FULL capture list."""
    result = {mode: list(descs) for mode, descs in original.items()}
    extras = tuple(factory(cg_mode=full_mode, num_tokens=n*q, num_reqs=n,
                           uniform_token_count=q, num_active_loras=0)
                   for n, q in CASES)
    if any(desc in result[full_mode] for desc in extras):
        raise RuntimeError('Uniform target graph case already exists')
    result[full_mode] += extras
    result[full_mode].sort(key=lambda d: d.num_tokens, reverse=True)
    return result, extras
