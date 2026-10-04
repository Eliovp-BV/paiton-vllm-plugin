"""DFlash drafter vocabulary modules on the meta device (run inside the serving image: needs vllm and torch).

The Qwen3.8 drafter checkpoint ships neither embed_tokens nor lm_head; load_dflash_model replaces both with the target
model's modules right after loading. Built on the GPU they cost 2 x 248,320 x 5,120 bf16 = 4.74 GiB at the load-phase
peak for nothing, so the overlay builds them on the meta device, materialises them only when a checkpoint ships them,
and refuses a drafter that still holds a meta tensor after sharing (a meta tensor reads as nothing, silently).
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402

COMPAT = Path(__file__).resolve().parents[1] / "paiton_vllm_plugin/_vendor/qwen38/compat"
MODEL = "vllm/model_executor/models/qwen3_dflash.py"
UTILS = "vllm/v1/worker/gpu/spec_decode/dflash/utils.py"


def _load(rel, name):
    # named inside the overlay's package so its relative imports resolve
    spec = importlib.util.spec_from_file_location(name, COMPAT / "overlays" / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model = _load(MODEL, "vllm.model_executor.models.paiton_test_qwen3_dflash")
utils = _load(UTILS, "vllm.v1.worker.gpu.spec_decode.dflash.paiton_test_utils")


def _vocab_module():
    module = nn.Module()
    module.weight = nn.Parameter(torch.empty(64, 8), requires_grad=False)
    module.weight.weight_loader = lambda param, loaded: param.data.copy_(loaded)
    module.weight.output_dim = 0
    return module


def test_shared_vocab_modules_are_built_on_meta():
    with model._paiton_vocab_context(True):
        shared = _vocab_module()
    with model._paiton_vocab_context(False):
        own = _vocab_module()
    assert shared.weight.is_meta and not own.weight.is_meta


def test_switch_builds_them_on_the_device(monkeypatch):
    monkeypatch.setattr(model, "_PAITON_META_VOCAB", False)
    with model._paiton_vocab_context(True):
        assert not _vocab_module().weight.is_meta


def test_a_checkpoint_with_its_own_vocab_loads_into_real_storage():
    with model._paiton_vocab_context(True):
        module = _vocab_module()
    model._paiton_materialize(module, torch.device("cpu"))
    weight = module.weight
    assert not weight.is_meta and weight.device.type == "cpu"
    assert isinstance(weight, nn.Parameter) and dict(module.named_parameters())["weight"] is weight
    assert weight.output_dim == 0                      # vLLM's loader attributes survive (to_empty drops them)
    loaded = torch.arange(64 * 8, dtype=weight.dtype).view(64, 8)
    weight.weight_loader(weight, loaded)
    assert torch.equal(weight, loaded)


def test_a_drafter_left_with_a_meta_tensor_is_refused():
    drafter = nn.Module()
    drafter.model = nn.Module()
    with model._paiton_vocab_context(True):
        drafter.model.embed_tokens = _vocab_module()
        drafter.lm_head = _vocab_module()
    with pytest.raises(RuntimeError, match="meta tensors"):
        utils.paiton_refuse_meta(drafter)
    target_embed, target_head = _vocab_module(), _vocab_module()
    del drafter.model.embed_tokens
    drafter.model.embed_tokens = target_embed
    with pytest.raises(RuntimeError, match="lm_head"):
        utils.paiton_refuse_meta(drafter)
    del drafter.lm_head
    drafter.lm_head = target_head
    utils.paiton_refuse_meta(drafter)


def test_load_dflash_model_checks_after_sharing():
    source = (COMPAT / "overlays" / UTILS).read_text()
    body = source.split("def load_dflash_model", 1)[1].split("\ndef ", 1)[0]
    assert body.index("dflash_model.lm_head = target_lm_head") < body.index("paiton_refuse_meta(dflash_model)")


@pytest.mark.parametrize("rel", [MODEL, UTILS])
def test_manifest_pins_the_overlays(rel):
    entry = json.loads((COMPAT / "compat_manifest.json").read_text())["files"]["/usr/local/lib/python3.12/dist-packages/"
                                                                                + rel]
    assert entry["overlay"] == "overlays/" + rel
    assert hashlib.sha256((COMPAT / entry["overlay"]).read_bytes()).hexdigest() == entry["overlay_sha256"]
