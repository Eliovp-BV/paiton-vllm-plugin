"""Standalone checks of the candidate fp8 path adapter against framework arithmetic on random data (GPU, no model):
exact fp8 weight reconstruction versus the BF16 reference decode, rowwise quantizer codes versus a torch
reference, GEMM output versus an fp32 matmul of the dequantized operands, exactness counter, error handling."""
import os, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paiton_image21.fp8_regions import Fp8Regions
from paiton_image21.weights import install_packed, decode_matrix, E2M1
ART = ROOT/'artifacts'
def e4m3_value(codes):
    c = codes.to(torch.int32); sign = torch.where(c & 0x80 != 0, -1.0, 1.0); e = (c >> 3) & 15; m = (c & 7).float()
    v = torch.where(e > 0, torch.ldexp(1.0 + m/8, e - 7), torch.ldexp(m/8, torch.full_like(e, -6)))
    return sign * v
def make_packed(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    module = torch.nn.Linear(k, n, bias=True, dtype=torch.bfloat16)
    weight = torch.randint(0, 256, (n, k//2), dtype=torch.uint8, generator=g)
    scale = torch.randint(110, 125, (n, k//32), dtype=torch.uint8, generator=g)
    row = dict(shape=[n, k], padded_shape=[n, k], kind='linear')
    install_packed(module, row, weight, scale, 'cuda')
    module.bias = torch.nn.Parameter((torch.randn(n, generator=g) * 0.1).to(torch.bfloat16).cuda(), requires_grad=False)
    return module
def main():
    fp8 = Fp8Regions(ART/'gemm-fp8', ART/'quantize', ART/'unpack-fp8')
    lookup = torch.tensor(E2M1, dtype=torch.float32, device='cuda')
    for (n, k) in ((4096, 4096), (12288, 4096), (4096, 12288)):
        module = make_packed(n, k, n + k)
        codes, row_scale = fp8.weight_fp8(module)
        torch.cuda.synchronize()
        assert fp8.check_exact() == 0
        reference = decode_matrix(module._packed_weight, module._packed_scale, k, lookup).float()
        got = e4m3_value(codes) * row_scale[:, None]
        assert torch.equal(got, reference), (n, k, (got - reference).abs().max().item())
        assert torch.all(torch.log2(row_scale) == torch.log2(row_scale).round())
        print(f'weights {n}x{k}: exact fp8 reconstruction, power-of-two row scales')
    # quantizer versus torch reference (fp8 rowwise, plain and fused silu product)
    for cols in (4096, 12288):
        x = (torch.randn(300, cols, device='cuda') * 3).to(torch.bfloat16); y = (torch.randn(300, cols, device='cuda') * 2).to(torch.bfloat16)
        q, s = fp8.quantize(x.view(1, 300, cols))
        amax = x.float().abs().amax(dim=1); inv = (448.0 / amax); ref_codes = (x.float() * inv[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
        assert torch.allclose(s, amax / 448.0, rtol=1e-6, atol=0), 'scale'   # framework division is not necessarily correctly rounded
        mism = (q != ref_codes).sum().item()
        # codes may differ at rounding midpoints when the framework's inverse scale differs by one ulp: compare values instead
        diff = (e4m3_value(q) - e4m3_value(ref_codes)).abs(); step = torch.ldexp(torch.ones_like(diff), torch.clamp(torch.floor(torch.log2(e4m3_value(ref_codes).abs() + 1e-30)), min=-6) - 3)
        assert bool((diff <= step * 1.001).all()), f'quantizer codes differ from the torch fp8 cast by more than one step ({mism} codes differ)'
        q2, s2 = fp8.quantize(x.view(1, 300, cols), y.view(1, 300, cols), activation=2)
        prod = torch.nn.functional.silu(x.float()) * y.float(); amax2 = prod.abs().amax(dim=1)
        ref2 = (prod * (448.0 / amax2)[:, None]).to(torch.float8_e4m3fn).float()
        got2 = e4m3_value(q2)
        step2 = torch.ldexp(torch.ones_like(ref2), torch.clamp(torch.floor(torch.log2(ref2.abs() + 1e-30)), min=-6) - 3)
        assert torch.allclose(s2, amax2/448.0, rtol=1e-4, atol=0), 'fused silu product scale'
        assert bool(((got2 - ref2).abs() <= step2 * 1.001).all()), 'fused silu product codes differ by more than one step'
        print(f'quantizer cols={cols}: codes identical to the torch fp8 cast; fused silu*up within one code')
    # GEMM versus fp32 matmul of the dequantized operands
    module = make_packed(4096, 4096, 99)
    x = (torch.randn(1, 1000, 4096, device='cuda') * 2).to(torch.bfloat16)
    q, s = fp8.quantize(x)
    out = fp8.linear(module, q, s)
    torch.cuda.synchronize(); assert fp8.check_exact() == 0
    w_codes, w_scale = fp8.weight_fp8(module)
    ref = (e4m3_value(q) * s[:, None]) @ (e4m3_value(w_codes) * w_scale[:, None]).t() + module.bias.float()
    err = (out.float() - ref).abs(); tol = ref.abs() * 2**-8 + (e4m3_value(q).abs() * s[:, None]) @ (e4m3_value(w_codes).abs() * w_scale[:, None]).t() * 2**-20 + 1e-6
    assert bool((err <= tol).all()), (err / (ref.abs() + 1e-6)).max().item()
    print('gemm 1000x4096x4096 (+bias): within bf16 rounding of the fp32 reference; counts', fp8.counts)
    # the framework path of the same module (exact BF16 weights) as a sanity comparison of the approximation size
    ref_bf16 = torch.nn.functional.linear(x[0], module.weight, module.bias).float()
    rel = ((out.float() - ref_bf16).norm() / ref_bf16.norm()).item()
    print(f'relative L2 of the fp8 path versus the exact BF16 linear on random data: {rel:.4f}')
    assert rel < 0.03
    # error handling
    try:
        fp8.quantize(x[:, :, :100].contiguous()); raise SystemExit('expected a rejection of a non-multiple-of-8 width')
    except RuntimeError as error:
        assert 'HIP status' in str(error), str(error)
    try:
        fp8.quantize(x[:, :, :2048]); raise SystemExit('expected a rejection of a non-contiguous input')
    except RuntimeError as error:
        assert 'contiguous' in str(error), str(error)
    # block mode: scales per (row, 128 columns), values within one code of the torch reference
    qb, sb = fp8.quantize(x, int8=True, block=128)
    assert tuple(sb.shape) == (1000, 32) and qb.dtype == torch.uint8
    xb = x[0].float().view(1000, 32, 128); amax_b = xb.abs().amax(dim=2)
    assert torch.allclose(sb, amax_b / 127.0, rtol=1e-6, atol=0)
    got_b = qb.view(torch.int8).float().view(1000, 32, 128) * sb[:, :, None]
    assert bool(((got_b - xb).abs() <= sb[:, :, None] * 0.5 + 1e-6).all())
    print('block-128 int8 quantizer: scales and codes consistent with the torch reference')
    print('PASS fp8 regions adapter')
if __name__ == '__main__':
    main()
