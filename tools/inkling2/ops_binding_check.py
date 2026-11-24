"""Check the plugin's binding of the native operator library on the per-operator fixtures.

Usage: PAITON_INK2_OPS=native PAITON_INK2_EXPERTS=native PAITON_INK2_OPS_SO=<inkling2_ops_gfx1201.so> \
       python3 ops_binding_check.py <compiler native/inkling2_ops dir> <fixtures/ops/runNN[/prompt]> <model safetensors> \
                                    [<folder of npz files with LM head inputs `final`>]

Runs the kernel piece's own comparisons (`check.py` and `ref.py` of the compiler component, numpy only) with torch
tensors through `paiton_vllm_plugin.inkling2.amd.native_ops` in place of the ctypes harness: the same rows, limits
and pass counts as `check.py` on the same fixtures. Then:
  - the convenience calls (`qkvr_project`, `controller_products`, `lm_head`, `linear_k_groups`) against the plain
    `linear` calls they are made of, bit for bit, and the projection switch `PAITON_INK2_PROJ` on every 8-bit
    format of the library;
  - the import rule of the native mode: with `amd/attention.py` replaced by a stub (the lead's file), importing
    `amd/loop.py` pulls in no module under `amd/ops/` and none under `vllm.models.inkling.amd`.
Host arrays go to the card and back through one pinned 32 MB buffer.
"""
import ctypes as C
import os
import sys
import types

import numpy as np
import torch

from paiton_vllm_plugin.inkling2.amd import native_ops

check_dir, fixtures, ckpt = sys.argv[1:4]
head_rows = sys.argv[4] if len(sys.argv) > 4 else None   # LM head inputs for the 8-bit head gates (ABI 3 library)
sys.path.insert(0, check_dir)
import check  # noqa: E402  (the compiler component's harness)

BF16, F32 = torch.bfloat16, torch.float32


class TorchOps:
    """The methods of `check.Gpu`, on torch tensors through `native_ops`."""

    def __init__(self) -> None:
        self.dev = torch.device("cuda:0")
        self.stage = torch.empty(32 << 20, dtype=torch.uint8).pin_memory()
        self.stage_np = self.stage.numpy()
        self.kept: dict[int, torch.Tensor] = {}  # weights that stay, by device pointer
        self.temp: dict[int, torch.Tensor] = {}  # until free()
        self.hip = self  # check.py frees kept weights with gpu.hip.hipFree

    def up(self, a: np.ndarray) -> torch.Tensor:
        a = np.ascontiguousarray(a)
        dtype = BF16 if a.dtype in (np.int16, np.uint16) else torch.from_numpy(np.empty(0, a.dtype)).dtype
        t = torch.empty(a.shape, dtype=dtype, device=self.dev)
        raw, dst = a.reshape(-1).view(np.uint8), t.view(-1).view(torch.uint8)
        for o in range(0, raw.size, self.stage_np.size):
            n = min(self.stage_np.size, raw.size - o)
            self.stage_np[:n] = raw[o:o + n]
            dst[o:o + n].copy_(self.stage[:n])
        return t

    def down(self, t: torch.Tensor) -> np.ndarray:
        assert t.is_contiguous()
        dtype = np.int16 if t.dtype == BF16 else torch.empty(0, dtype=t.dtype).numpy().dtype
        out = np.empty(tuple(t.shape), dtype)
        raw, src = out.reshape(-1).view(np.uint8), t.view(-1).view(torch.uint8)
        for o in range(0, raw.size, self.stage_np.size):
            n = min(self.stage_np.size, raw.size - o)
            self.stage[:n].copy_(src[o:o + n])
            raw[o:o + n] = self.stage_np[:n]
        return out

    def opt(self, a):
        return None if a is None else self.up(a)

    def keep(self, a):
        t = self.up(a)
        self.kept[t.data_ptr()] = t
        return C.c_void_p(t.data_ptr())

    def put(self, a):
        t = self.up(a)
        self.temp[t.data_ptr()] = t
        return C.c_void_p(t.data_ptr())

    def hipFree(self, p) -> None:
        self.kept.pop(p.value, None)

    def free(self) -> None:
        self.temp.clear()

    def weight(self, p, k: int) -> tuple[torch.Tensor, int]:
        """The uploaded weight tensor a device pointer falls into, and the row it points at."""
        for base, t in (*self.kept.items(), *self.temp.items()):
            if base <= p.value < base + t.numel() * 2:
                return t, (p.value - base) // (2 * k)
        raise KeyError("unknown weight pointer")

    def rms_norm(self, x, weight, eps, repeat=1):
        xt, wt = self.up(x), self.up(weight)
        if weight.shape[0] == 1:  # the port's call: one weight row, 2-D result, bf16 asked for
            return self.down(native_ops.rms_norm(xt, wt[0], eps, BF16))[None]
        return self.down(native_ops.rms_norm(xt, wt, eps, BF16))

    def embed_norm(self, table, vocab, ids, weight, eps, repeat=1):
        t, _ = self.weight(table, 2304)
        assert t.shape[0] == vocab
        return self.down(native_ops.embed_norm(self.up(ids), t, self.up(weight)[0], eps))   # int32 and int64 ids

    def lm_head(self, x, w, n, soft_cap=0.0, repeat=1):
        t, row = self.weight(w, x.shape[-1])
        assert row == 0
        return self.down(native_ops.lm_head(self.up(x), t, n, soft_cap))

    def head_rows(self, fmt, w, n, k):
        t, row = self.weight(w, k)
        return native_ops.head_rows(t[row:row + n], fmt)

    def head_get(self, handle, lo, hi):
        return self.down(handle[0][lo:hi]), self.down(handle[1][lo:hi])

    def head_free(self, handle) -> None:
        pass

    def linear_rows(self, fmt, x, handle, lo, hi, out_f32, repeat=1):
        return self.down(native_ops.linear_rows(self.up(x), *handle, fmt, F32 if out_f32 else BF16, rows=slice(lo, hi)))

    def lm_head_rows(self, fmt, x, handle, soft_cap=0.0, repeat=1):
        return self.down(native_ops.lm_head_rows(self.up(x), *handle, soft_cap, fmt))

    def pair_rms_norm(self, dest, message, repeat=1):
        dt, mt = self.up(dest), self.up(message)
        if dt.shape[0] == 1 and mt.shape[0] > 1:  # the loop's broadcast over the streams
            dt = dt.expand(mt.shape[0], -1, -1)
        return self.down(native_ops.pair_rms_norm(dt, mt))

    def blend_rms_norm(self, dest, message, gain_forget, weight, eps, repeat=1):
        return self.down(native_ops.blend_rms_norm(self.up(dest), self.up(message), self.up(gain_forget),
                                                   self.up(weight), eps))

    def attn_res(self, stack, last, addend, queries, norm_weight, eps, norm_eps, normalize=True, repeat=1):
        return self.down(native_ops.attn_res(self.up(stack), self.up(last), self.opt(addend), self.up(queries),
                                             self.opt(norm_weight), norm_eps, normalize, eps))

    def seq_metadata(self, query_start_loc, actual, padded):
        a, b = (torch.empty(padded, dtype=torch.int32, device=self.dev) for _ in range(2))
        native_ops.seq_metadata(self.up(query_start_loc.astype(np.int32)), len(query_start_loc) - 1, actual, a, b,
                                padded)
        return self.down(a), self.down(b)

    def short_conv(self, x, weight, cache, positions, seq_idx, slots, qstart, block_table, off, width, addend=None,
                   residual=None, repeat=1, dyn=None):
        ct = self.up(cache)
        paging = (ct, self.up(positions.astype(np.int64)), self.up(block_table.astype(np.int32)),
                  self.up(seq_idx.astype(np.int32)), self.up(slots.astype(np.int64)), self.up(qstart.astype(np.int32)))
        if dyn is not None:
            out = native_ops.short_conv_dynamic(self.up(x), self.up(weight), self.up(dyn), *paging, off, width)
        else:
            res = self.opt(residual)  # the model passes the residual as the output buffer
            out = native_ops.short_conv(self.up(x), self.up(weight), *paging, off, width, self.opt(addend), res, res)
        return self.down(out), self.down(ct)

    def linear(self, x, w, n, out_f32, soft_cap=0.0, repeat=1):
        xt = self.up(x)
        wt, row = self.weight(w, x.shape[-1])
        dtype = F32 if out_f32 else BF16
        if wt.dim() == 2:
            return self.down(native_ops.linear(xt[0], wt, dtype, soft_cap, rows=slice(row, row + n)))[None]
        assert row == 0 and wt.shape[-2] == n
        return self.down(native_ops.linear(xt, wt, dtype, soft_cap))


def extras(ops: TorchOps) -> bool:
    """The convenience calls against the plain `linear` calls they stand for."""
    g = torch.Generator().manual_seed(3)
    rand = lambda *shape: ops.up(torch.randn(shape, generator=g).to(BF16).view(torch.int16).numpy())  # noqa: E731
    same = lambda a, b: torch.equal(a.contiguous().view(torch.int16 if a.dtype == BF16 else torch.int32),  # noqa: E731
                                    b.contiguous().view(torch.int16 if b.dtype == BF16 else torch.int32))
    ok = True
    for tokens in (1, 40):
        x, w, splits = [rand(tokens, 2304) for _ in range(3)], rand(8600, 2304), [3072, 384, 3072, 24, 1024, 1024]
        got = native_ops.qkvr_project(*x, w, splits)
        full = [native_ops.linear(xi, w) for xi in x]  # each input through the whole matrix
        cols = (0, 3072, 3456, 6528, 6552, 7576, 8600)
        want = [full[(0, 0, 0, 0, 1, 2)[i]][:, cols[i]:cols[i + 1]] for i in range(6)]
        good = all(same(a, b) for a, b in zip(got, want)) and got[0].stride(0) == 6552 and got[4].is_contiguous()
        pair, down, up = rand(4, tokens, 4608), rand(4, 64, 4608), rand(4, 4608, 64)
        good &= same(native_ops.controller_products(pair, down, up),
                     native_ops.linear(native_ops.linear(pair, down), up, F32))
        head = rand(2600, 2304)
        logits = native_ops.lm_head(x[0], head, 2570, 30.0)
        ref_logits = native_ops.linear(x[0], head, F32, 30.0)[:, :2570]   # the LM head kernel sums in another order
        good &= tuple(logits.shape) == (tokens, 2570) and bool((logits - ref_logits).abs().max() <= 3e-4)
        good &= same(native_ops.linear_k_groups(x[0], head, BF16), native_ops.linear(x[0], head))
        empty = native_ops.linear(x[0][:0], head)
        good &= tuple(empty.shape) == (0, 2600)
        print(f"convenience calls, {tokens} rows: {'ok' if good else 'FAIL'}", flush=True)
        ok &= good
    torch.cuda.synchronize()
    return ok


def proj_switch(ops: TorchOps) -> bool:
    """PAITON_INK2_PROJ: `qkvr_project` and `o_project` on 8-bit rows equal `linear_rows` on the rows of the weight."""
    g = torch.Generator().manual_seed(4)
    rand = lambda *shape: ops.up(torch.randn(shape, generator=g).to(BF16).view(torch.int16).numpy())  # noqa: E731
    same = lambda a, b: torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))  # noqa: E731
    x, w, wo, y = [rand(40, 2304) for _ in range(3)], rand(8600, 2304), rand(2304, 3072), rand(40, 3072)
    cols, ok = (0, 3072, 3456, 6528, 6552, 7576, 8600), True
    for fmt in native_ops.proj_formats():
        native_ops.PROJ = fmt   # the switch is read at import; set it for this test
        native_ops._proj_rows.clear()
        try:
            got = native_ops.qkvr_project(*x, w, [3072, 384, 3072, 24, 1024, 1024])
            codes, scale = native_ops.head_rows(w, fmt)
            full = [native_ops.linear_rows(xi, codes, scale, fmt) for xi in x]
            good = all(same(got[i], full[(0, 0, 0, 0, 1, 2)[i]][:, cols[i]:cols[i + 1]]) for i in range(6))
            out = native_ops.o_project(y, wo)
            good &= same(out, native_ops.linear_rows(y, *native_ops.head_rows(wo, fmt), fmt))
            good &= not same(out, native_ops.linear(y, wo))   # it did run on the 8-bit rows
        finally:
            native_ops.PROJ = "bf16"
            native_ops._proj_rows.clear()
        print(f"projection switch, {fmt} rows: {'ok' if good else 'FAIL'}", flush=True)
        ok &= good
    torch.cuda.synchronize()
    return ok


def import_rule() -> bool:
    name = "paiton_vllm_plugin.inkling2.amd.attention"
    stub = types.ModuleType(name)
    stub.Inkling2Attention = type("Inkling2Attention", (torch.nn.Module,), {})
    sys.modules[name] = stub
    import paiton_vllm_plugin.inkling2.amd.loop  # noqa: F401
    bad = sorted(m for m in sys.modules if m.startswith(("paiton_vllm_plugin.inkling2.amd.ops.", "vllm.models.inkling.amd"))
                 or m in ("paiton_vllm_plugin.inkling2.amd.ops", "paiton_vllm_plugin.inkling2.amd.moe",
                          "paiton_vllm_plugin.inkling2.amd.vision", "paiton_vllm_plugin.inkling2.amd.dspark"))
    print("import rule (attention.py stubbed):", "ok" if not bad else f"FAIL, imported {bad}", flush=True)
    return not bad


if __name__ == "__main__":
    assert os.environ.get("PAITON_INK2_OPS") == "native", "set PAITON_INK2_OPS=native"
    ops = TorchOps()
    ops.head_formats = native_ops.head_formats()   # none with an ABI 2 library: check.py then leaves those rows out
    ops.proj_formats = native_ops.proj_formats()
    failed = check.run_checks(ops, fixtures, ckpt, head_rows)
    torch.cuda.synchronize()
    failed |= not extras(ops)
    failed |= not proj_switch(ops)
    failed |= not import_rule()
    print("OPS BINDING CHECK", "FAILED" if failed else "PASSED")
    sys.exit(1 if failed else 0)
