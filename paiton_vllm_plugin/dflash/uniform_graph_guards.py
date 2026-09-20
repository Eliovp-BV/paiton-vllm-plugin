"""Source guards for decode classification and the captured state layout."""
import hashlib
import json

# Original source and active compatibility overlay, respectively.
PINS = {
    'vllm/v1/worker/gpu/model_runner.py': (
        'd7bae5b05a2a9f02c1ba29f0b4b32434cee43c3b369d3985505d3f8ed89b461b',
        'ad78894ad3e74d10cea04aa9d8f942f5e8c61006738a83c76600469cffa1155c'),
    'vllm/v1/worker/utils.py': (
        '655c4ad9d87d3b0c559e4ad3fffe2478476f9f1472c20d4341b7a6c9ef7f2c33',
        '655c4ad9d87d3b0c559e4ad3fffe2478476f9f1472c20d4341b7a6c9ef7f2c33'),
    'vllm/v1/attention/backends/gdn_attn.py': (
        'c65552d9aad86472544033d44ad8a872221a83e6b60ba9918cc049ab0c580c7c',
        '79b9c623484576208f72f2228af8d507c96bd176f21560f3d306c5f709a40dff'),
}


def validate_runtime_sources(base):
    runtime=(base/'paiton_runtime_compat').resolve(strict=True)
    entries=json.loads((runtime/'compat_manifest.json').read_text())['files']
    for relative,(original_sha,effective_sha) in PINS.items():
        path=base/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=original_sha:
            raise RuntimeError('Uniform graph original source changed: '+relative)
        entry=entries.get(str(path))
        if original_sha==effective_sha:
            if entry is not None:
                raise RuntimeError('Unexpected uniform graph runtime overlay: '+relative)
            continue
        if (entry is None or entry['original_sha256']!=original_sha
                or entry['overlay_sha256']!=effective_sha):
            raise RuntimeError('Uniform graph overlay metadata changed: '+relative)
        overlay=(runtime/entry['overlay']).resolve(strict=True)
        if not overlay.is_relative_to(runtime):
            raise RuntimeError('Uniform graph overlay escaped its package')
        if hashlib.sha256(overlay.read_bytes()).hexdigest()!=effective_sha:
            raise RuntimeError('Uniform graph active source changed: '+relative)


def runtime_scope(config):
    cache = getattr(config, 'cache_config', None)
    scheduler = getattr(config, 'scheduler_config', None)
    spec = getattr(config, 'speculative_config', None)
    model = getattr(config, 'model_config', None)
    return (
        getattr(cache, 'enable_prefix_caching', None) is False
        and getattr(scheduler, 'async_scheduling', None) is False
        and getattr(spec, 'draft_sample_method', None) == 'greedy'
        and getattr(spec, 'disable_padded_drafter_batch', None) is True
        and getattr(model, 'max_model_len', None) == 65536
    )
