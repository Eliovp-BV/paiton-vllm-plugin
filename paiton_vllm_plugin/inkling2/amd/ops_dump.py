# SPDX-License-Identifier: Apache-2.0
"""Per-operator fixtures from the bring-up port, for the native operators.

Bring-up tool: nothing here runs unless PAITON_INK2_OPS_DUMP=<dir> is set. It
then wraps the small operators of the looped model and saves every call made
during the chosen visits of the first forwards as
`<dir>/f<forward>-<layer>p<pass>-<n>-<op>.npz`:

  args.<name>     every argument as passed (before the call)
  out, out.<i>    the return value(s)
  after.<name>    arguments the operator writes in place, after the call
  cache.<name>.blocks / .before / .after
                  for cache-writing operators: the cache blocks this call's
                  tokens use (block ids, contents before and after)
  __meta__        JSON: op, forward, layer, pass, scalars, and for every array
                  its torch dtype (bf16 is stored as int16 bits, fp8 as uint8)

PAITON_INK2_OPS_VISITS  comma list of `layer.pass`, plus `pre` (embedding) and
                        `post` (final mix, norm and LM head).
                        Default: pre,0.0,5.0,6.0,6.1,6.2,11.1,18.0,23.0,post
PAITON_INK2_OPS_FORWARDS  number of forwards to record (default 3: with one
                        sequence, the prefill and two decode steps).
PAITON_INK2_OPS_FIRST   first forward to record (default 1: vLLM 0.32 runs one
                        profiling forward of max_num_batched_tokens dummy
                        tokens without caches at start-up, forward 0).
"""

import inspect
import json
import os

import numpy as np
import torch

DIR = os.environ.get("PAITON_INK2_OPS_DUMP")
VISITS = set(
    os.environ.get(
        "PAITON_INK2_OPS_VISITS", "pre,0.0,5.0,6.0,6.1,6.2,11.1,18.0,23.0,post"
    ).split(",")
)
FIRST = int(os.environ.get("PAITON_INK2_OPS_FIRST", "1"))
FORWARDS = int(os.environ.get("PAITON_INK2_OPS_FORWARDS", "3"))

forward = -1  # index of the running forward, set by the loop model
where = ""  # "pre", "<layer>.<pass>" or "post", set by the loop model
_calls = 0

# Arguments written in place, and (cache, slot mapping, dimension that holds
# the tokens of a block) of cache writers: the conv state is (blocks, heads,
# tokens, width), the key and value views are (blocks, tokens, heads, width).
_IN_PLACE = {
    "short_conv": ("out", "residual"),
    "attention_prologue": ("q_out", "rel_out"),
    "fused_add_rms_norm": ("x", "residual"),
}
_CACHES = {
    "short_conv": (("cache", "slot_mapping", 2),),
    "attention_prologue": (
        ("cache", "slot_mapping", 2),
        ("key_cache", "kv_slot_mapping", 1),
        ("value_cache", "kv_slot_mapping", 1),
    ),
}


def _put(arrays: dict, meta: dict, name: str, value) -> None:
    if isinstance(value, torch.Tensor):
        t = value.detach().contiguous().cpu()
        meta["dtypes"][name] = str(t.dtype).removeprefix("torch.")
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        elif not t.dtype.is_floating_point or t.dtype in (torch.float32, torch.float16):
            pass
        else:  # float8 formats
            t = t.view(torch.uint8)
        arrays[name] = t.numpy()
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _put(arrays, meta, f"{name}.{i}", item)
    elif isinstance(value, slice):
        meta["scalars"][name] = [value.start, value.stop, value.step]
    else:
        meta["scalars"][name] = (
            value if isinstance(value, (int, float, bool, str, type(None))) else repr(value)
        )


def _used_blocks(
    cache: torch.Tensor, slots: torch.Tensor, table: torch.Tensor, block_dim: int
):
    """Blocks of `cache` the call can touch: those of the written slots and
    every valid entry of the block table."""
    written = slots[slots >= 0] // cache.shape[block_dim]
    listed = table[(table >= 0) & (table < cache.shape[0])].flatten()
    return torch.unique(torch.cat((written.long(), listed.long())))


def record(op: str, args: dict, call):
    """Run `call()` and save the fixture of this call when its visit is chosen."""
    global _calls
    if DIR is None or not FIRST <= forward < FIRST + FORWARDS or where not in VISITS:
        return call()
    arrays: dict = {}
    meta: dict = {"op": op, "forward": forward, "where": where, "dtypes": {}, "scalars": {}}
    caches = {}
    for cache, slots, block_dim in _CACHES.get(op, ()):
        ids = _used_blocks(args[cache], args[slots], args["block_table"], block_dim)
        caches[cache] = ids
        _put(arrays, meta, f"cache.{cache}.blocks", ids)
        _put(arrays, meta, f"cache.{cache}.before", args[cache][ids])
        meta["scalars"][f"cache.{cache}.shape"] = list(args[cache].shape)
    for name, value in args.items():
        if name not in caches:
            _put(arrays, meta, f"args.{name}", value)
    out = call()
    _put(arrays, meta, "out", out)
    for name in _IN_PLACE.get(op, ()):
        if args.get(name) is not None:
            _put(arrays, meta, f"after.{name}", args[name])
    for cache, ids in caches.items():
        _put(arrays, meta, f"cache.{cache}.after", args[cache][ids])
    tag = where.replace(".", "p")
    np.savez(f"{DIR}/f{forward}-{tag}-{_calls:04d}-{op}.npz", __meta__=json.dumps(meta), **arrays)
    _calls += 1
    return out


def _wrap(fn, op: str | None = None):
    sig, op = inspect.signature(fn), op or fn.__name__

    def wrapped(*args, **kwargs):
        if DIR is None or not FIRST <= forward < FIRST + FORWARDS or where not in VISITS:
            return fn(*args, **kwargs)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        named = {k: v for k, v in bound.arguments.items() if k != "self"}
        return record(op, named, lambda: fn(*args, **kwargs))

    return wrapped


def install() -> None:
    """Wrap the operators where the model modules imported them."""
    from . import attention, loop, model, sconv

    for module, names in (
        (model, ("rms_norm", "attn_res")),
        (loop, ("attn_res", "blend_rms_norm", "pair_rms_norm")),
        (sconv, ("short_conv", "linear_k_groups")),
        (attention, ("attention_prologue",)),
    ):
        for name in names:
            setattr(module, name, _wrap(getattr(module, name)))

    project = attention.Inkling2Attention._project

    def recorded_project(self, x_q, x_k, x_v):
        args = {
            "x_q": x_q,
            "x_k": x_k,
            "x_v": x_v,
            "shared_input": x_k is x_q and x_v is x_q,
            "split_sizes": repr(list(self.split_sizes)),
            # 37 MB per layer: once, with the prefill forward's calls.
            "weight": self.qkvr.weight if forward == FIRST else None,
        }
        return record("qkvr_linear", args, lambda: project(self, x_q, x_k, x_v))

    attention.Inkling2Attention._project = recorded_project
