"""CPU checks of the Inkling 2 bring-up experts (PyTorch only, no vLLM, no GPU)."""
import types

import pytest

torch = pytest.importorskip("torch")
eb = pytest.importorskip("paiton_vllm_plugin.inkling2.amd.experts_bringup")


def test_grids_and_dequant():
    g = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6.0])
    assert torch.equal(eb.e2m1(torch.cat([g, -g])), torch.cat([g, -g]))
    assert eb.e2m1(torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 9.0])).tolist() == [0, 1, 1, 2, 2, 4, 4, 6]
    assert eb.e4m3(torch.tensor([448.0, 500.0, 1.0625, 1.1875, 0.0])).tolist() == [448, 448, 1.0, 1.25, 0]
    # codes 0..15, even element in the low nibble, block scale 2.0
    packed = torch.tensor([[(2 * i + 1) << 4 | (2 * i) for i in range(8)]], dtype=torch.uint8)
    two = torch.tensor([[2.0]]).to(torch.float8_e4m3fn)
    assert eb.dequant_nvfp4(packed, two)[0].tolist() == [0, 1, 2, 3, 4, 6, 8, 12, -0.0, -1, -2, -3, -4, -6, -8, -12]
    values = torch.tensor([[1.5] * 32]).to(torch.float8_e4m3fn)
    assert eb.dequant_mxfp8(values, torch.tensor([[128]], dtype=torch.uint8))[0].tolist() == [3.0] * 32


def test_activation_quantizers():
    torch.manual_seed(0)
    x = torch.randn(64, 2304) * 3
    for fq, tol in ((eb.fq_nvfp4, 0.12), (eb.fq_mxfp8, 0.04)):
        y = fq(x)
        assert torch.isfinite(y).all() and ((y - x).norm() / x.norm()).item() < tol
        assert fq(torch.zeros(2, 32)).abs().sum() == 0
        assert torch.equal(fq(y), y) or fq is eb.fq_nvfp4  # MXFP8 output is a fixed point


@pytest.mark.parametrize("act_quant", [False, True])
def test_experts_forward_matches_loop(act_quant):
    torch.manual_seed(1)
    cfg = types.SimpleNamespace(mlp_gate_softcap=4.0, mlp_up_softcap=0.0, n_routed_experts=4, hidden_size=64,
                                intermediate_size=32)
    ex = eb.Inkling2BringupExperts(cfg, "mxfp8", act_quant, True)
    ex.w13_weight.data = torch.randn(4, 64, 64).to(torch.float8_e4m3fn)
    ex.w13_weight_scale.data = torch.full((4, 64, 2), 127, dtype=torch.uint8)
    ex.w2_weight.data = torch.randn(4, 64, 32).to(torch.float8_e4m3fn)
    ex.w2_weight_scale.data = torch.full((4, 64, 1), 126, dtype=torch.uint8)
    w13, w2 = ex.w13_weight.data.float(), ex.w2_weight.data.float() * 0.5
    ex.process_weights_after_loading()
    assert torch.equal(ex.w13, w13) and torch.equal(ex.w2, w2) and ex.w13_weight.numel() == 0
    x = torch.randn(5, 64).to(torch.bfloat16)
    idx = torch.tensor([[0, 1], [1, 2], [3, 0], [2, 3], [1, 0]])
    wt = torch.rand(5, 2)
    bf16 = torch.bfloat16
    ref = torch.zeros(5, 64)
    for t in range(5):
        for j in range(2):
            e, row = idx[t, j], x[t : t + 1]
            if act_quant:  # fp32 GEMMs on the quantized inputs
                gate_up = (eb.fq_mxfp8(row.float()) @ w13[e].T).to(bf16).float()
            else:  # bf16 GEMMs
                gate_up = torch.nn.functional.linear(row, w13[e].to(bf16)).float()
            gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
            h = (torch.tanh(gate / 4.0) * 4.0 * torch.sigmoid(gate) * up).to(bf16)
            if act_quant:
                ref[t] += (eb.fq_mxfp8(h.float() * wt[t, j]) @ w2[e].T)[0]
            else:
                ref[t] += torch.nn.functional.linear(h, w2[e].to(bf16)).float()[0] * wt[t, j]
    assert torch.allclose(ex(x, idx, wt).float(), ref.to(bf16).float(), rtol=1e-2, atol=1e-2)
