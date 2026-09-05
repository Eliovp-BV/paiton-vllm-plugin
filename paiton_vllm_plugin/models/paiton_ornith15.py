"""vLLM wrapper for the Paiton Ornith 1.5 MXFP4 text backbone."""

from collections.abc import Iterable
import json
from pathlib import Path

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory

from paiton_vllm_plugin.artifact_manifest import manifest_path_for
from paiton_vllm_plugin.models.artifact_resolver import resolve_artifact_dir
from paiton_vllm_plugin.models.model_path import resolve_model_so_path
from paiton_vllm_plugin.models.paiton_qwen38 import (
    PaitonQwen38ForCausalLM,
    PaitonQwen38GDNCacheLayer,
)
from paiton_vllm_plugin.runtime.core import Model
from paiton_vllm_plugin.runtime.core.utils.ornith_memory import (
    preflight_ornith_memory,
)
from paiton_vllm_plugin.runtime.core.utils.ornith_mxfp4_loader import (
    OrnithMxfp4StreamingLoader,
    ornith_specs_from_manifest,
)
from paiton_vllm_plugin.runtime.core.utils.qwen38_loader import (
    configure_qwen38_cache_contract,
)
from paiton_vllm_plugin.vllm_compat import Attention, AttentionType


class PaitonOrnith15ForCausalLM(PaitonQwen38ForCausalLM):
    """Embedding/logits shell around the compiled Ornith text backbone."""

    supports_eagle3 = True
    # SupportsEagle3 is a runtime-checkable structural protocol. These two
    # attributes are inherited from its SupportsEagleBase contract and must be
    # present even though the target owns the ordinary target LM head and
    # embeddings rather than separate draft-only copies.
    has_own_lm_head = False
    has_own_embed_tokens = False
    # The compiled backbone has no Python rotary module for vLLM's DFlash
    # detector to inspect. Qwen3.5 uses NeoX-style rotary layout.
    is_neox_style = True
    DFLASH_AUX_HIDDEN_STATE_LAYERS = (2, 7, 12, 17, 23, 28, 33, 38)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        if prefix:
            raise ValueError("Paiton Ornith does not support pipeline prefixes")
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Paiton Ornith requires TP=1")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Paiton Ornith requires PP=1")
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None:
            if speculative_config.method != "dflash":
                raise ValueError("Paiton Ornith supports only DFlash speculation")
            if speculative_config.num_speculative_tokens != 16:
                raise ValueError("Paiton Ornith DFlash requires 16 speculative tokens")
        configure_qwen38_cache_contract(
            vllm_config.cache_config, resolve_auto=True
        )

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_text_config
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.dtype = vllm_config.model_config.dtype
        if self.dtype is not torch.bfloat16:
            raise ValueError("Paiton Ornith requires BF16 activations")

        model_ref = vllm_config.model_config.model
        self.model_path = resolve_artifact_dir(
            model_ref,
            revision=vllm_config.model_config.revision,
            token=vllm_config.model_config.hf_token,
            download_dir=vllm_config.load_config.download_dir,
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.model_so_path = resolve_model_so_path(
            self.model_path,
            Path(model_ref).name,
            self.tp_size,
            max_input_tokens=max_tokens,
            decode_partition_size=getattr(self.config, "decode_partition_size", None),
        )
        with manifest_path_for(self.model_so_path).open(encoding="utf-8") as source:
            self.manifest = json.load(source)
        (
            manifest_layers,
            manifest_layer_types,
            _,
            expected_constants,
        ) = ornith_specs_from_manifest(self.manifest)
        self.contract = self.manifest["paiton_ornith15_contract"]
        if speculative_config is not None:
            if int(self.contract["version"]) != 11:
                raise ValueError("Paiton Ornith DFlash requires a contract v11 artifact")
            expected_metadata_layers = tuple(
                index
                for index, layer_type in enumerate(self.config.layer_types)
                if layer_type == "full_attention"
            )
            if (
                self.contract.get("dflash_full_attention_metadata") != "per_layer"
                or tuple(
                    self.contract.get("dflash_full_attention_metadata_layers", ())
                )
                != expected_metadata_layers
            ):
                raise ValueError(
                    "Paiton Ornith DFlash requires per-layer full-attention metadata"
                )
            self._validate_dflash_draft_config(speculative_config)
        self._dflash_output_names = tuple(
            self.contract.get("dflash_aux_hidden_state_outputs", ())
        )
        self._aux_hidden_state_layers: tuple[int, ...] = ()
        self.memory_estimate = preflight_ornith_memory(
            self.manifest,
            hybrid_cache_reservation_bytes=(
                vllm_config.cache_config.kv_cache_memory_bytes
            ),
        )
        self.num_layers = int(self.contract["num_hidden_layers"])
        if self.num_layers != manifest_layers:
            raise ValueError("Ornith manifest layer count is internally inconsistent")
        if self.num_layers != self.config.num_hidden_layers:
            raise ValueError("Ornith artifact/config layer-count mismatch")
        if tuple(self.config.layer_types) != manifest_layer_types:
            raise ValueError("Ornith artifact/config layer schedule mismatch")
        if max_tokens > int(self.contract["max_num_batched_tokens"]):
            raise ValueError("vLLM token budget exceeds the Ornith artifact contract")
        if vllm_config.model_config.max_model_len > int(
            self.contract["max_context_length"]
        ):
            raise ValueError("vLLM max model length exceeds the Ornith artifact contract")
        if vllm_config.cache_config.block_size != int(
            self.contract["kv_cache_block_size"]
        ):
            raise ValueError("vLLM block size does not match the Ornith artifact")

        self.compiled_model = Model(str(self.model_so_path))
        compiled_output_names = set(
            self.compiled_model.get_output_name_to_index_map()
        )
        expected_output_names = {"hidden_states", *self._dflash_output_names}
        if compiled_output_names != expected_output_names:
            raise ValueError(
                "Ornith runtime/manifest output mismatch: "
                f"missing={sorted(expected_output_names - compiled_output_names)}, "
                f"extra={sorted(compiled_output_names - expected_output_names)}"
            )
        runtime_constants = set(self.compiled_model.get_constant_names())
        if runtime_constants != set(expected_constants):
            raise ValueError(
                "Ornith runtime/manifest constant mismatch: "
                f"missing={sorted(set(expected_constants) - runtime_constants)[:8]}, "
                f"extra={sorted(runtime_constants - set(expected_constants))[:8]}"
            )
        self.compiled_input_names = set(
            self.compiled_model.get_input_name_to_index_map()
        )
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=torch.bfloat16,
            prefix="model.embed_tokens",
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=torch.bfloat16,
            prefix="lm_head",
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_size
        )

        self.layer_types = manifest_layer_types
        static_context = {}
        self.cache_layers = nn.ModuleDict()
        for index, layer_type in enumerate(self.layer_types):
            if layer_type == "linear_attention":
                key = f"model.layers.{index}.linear_attn"
                layer = PaitonQwen38GDNCacheLayer(
                    self.config, vllm_config, prefix=key
                )
            else:
                key = f"model.layers.{index}.self_attn"
                layer = Attention(
                    num_heads=self.config.num_attention_heads,
                    head_size=self.config.head_dim,
                    scale=self.config.head_dim**-0.5,
                    num_kv_heads=self.config.num_key_value_heads,
                    cache_config=vllm_config.cache_config,
                    quant_config=None,
                    prefix=key,
                    attn_type=AttentionType.DECODER,
                )
            self.cache_layers[str(index)] = layer
            static_context[key] = layer
        vllm_config.compilation_config.static_forward_context = static_context
        self._dummy_inputs = {}
        self._stride_inputs = {}
        self._metadata_inputs = {}
        self._metadata_trace_records = []

    @staticmethod
    def _validate_dflash_draft_config(speculative_config: object) -> None:
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        hf_config = getattr(draft_model_config, "hf_config", None)
        if hf_config is None:
            raise ValueError("Paiton Ornith DFlash requires a resolved draft config")
        # vLLM resolves DFlash checkpoints through EAGLEConfig, whose public
        # model_type is "eagle" while the original Qwen3 config is retained in
        # its ``model`` member. Validate the checkpoint-native config so the
        # wrapper cannot make a compatible draft look incompatible.
        source_hf_config = getattr(hf_config, "model", None) or hf_config
        if tuple(getattr(source_hf_config, "architectures", ())) != (
            "DFlashDraftModel",
        ):
            raise ValueError("Paiton Ornith requires the bundled DFlashDraftModel")
        exact = {
            "model_type": "qwen3",
            "num_hidden_layers": 6,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "sliding_window": 4096,
        }
        for name, expected in exact.items():
            if getattr(source_hf_config, name, None) != expected:
                raise ValueError(
                    f"Paiton Ornith DFlash draft {name} must be {expected!r}"
                )
        if tuple(getattr(source_hf_config, "layer_types", ())) != (
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ):
            raise ValueError("Paiton Ornith DFlash draft layer schedule is incompatible")
        dflash = getattr(source_hf_config, "dflash_config", None)
        if not isinstance(dflash, dict):
            raise ValueError("Paiton Ornith DFlash draft metadata is missing")
        if dflash.get("block_size") != 16 or dflash.get("target_layer_ids") != [
            1,
            6,
            11,
            16,
            22,
            27,
            32,
            37,
        ]:
            raise ValueError("Paiton Ornith DFlash draft metadata is incompatible")

    def _compiled_gdn_metadata(
        self,
        metadata: object,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if int(self.contract["version"]) != 11:
            return super()._compiled_gdn_metadata(metadata, device)
        num_spec_decodes = int(getattr(metadata, "num_spec_decodes", 0))
        if num_spec_decodes:
            if (
                num_spec_decodes != 1
                or int(getattr(metadata, "num_prefills", 0)) != 0
                or int(getattr(metadata, "num_decodes", 0)) != 0
            ):
                raise ValueError(
                    "Paiton Ornith DFlash requires one isolated speculative request"
                )
            query_starts = getattr(metadata, "spec_query_start_loc", None)
            state_indices = getattr(metadata, "spec_state_indices_tensor", None)
            accepted = getattr(metadata, "num_accepted_tokens", None)
            if query_starts is None or state_indices is None or accepted is None:
                raise RuntimeError("vLLM did not provide complete DFlash GDN metadata")
            if state_indices.ndim != 2 or state_indices.shape != (1, 17):
                raise RuntimeError(
                    "vLLM DFlash GDN state indices must have shape [1, 17]"
                )
            if tuple(accepted.shape) != (1,):
                raise RuntimeError("vLLM DFlash acceptance metadata must have shape [1]")
            return (
                query_starts,
                state_indices,
                getattr(metadata, "has_initial_state", None),
                accepted,
            )
        query_starts = getattr(metadata, "non_spec_query_start_loc", None)
        state_indices = getattr(metadata, "non_spec_state_indices_tensor", None)
        if query_starts is None or state_indices is None:
            raise RuntimeError("vLLM did not provide non-speculative GDN state metadata")
        if state_indices.ndim != 1:
            raise RuntimeError("vLLM non-speculative GDN state indices must be rank one")
        accepted = self._int32_ones_input(
            "num_accepted_tokens", state_indices.shape[0], device
        )
        return (
            query_starts,
            state_indices.unsqueeze(-1),
            getattr(metadata, "has_initial_state", None),
            accepted,
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        expected = self.DFLASH_AUX_HIDDEN_STATE_LAYERS
        if int(self.contract["version"]) != 11:
            raise ValueError(
                "Ornith auxiliary hidden states require a DFlash contract v11 artifact"
            )
        if tuple(layers) != expected:
            raise ValueError(
                f"Ornith DFlash requires auxiliary layers {expected}, got {tuple(layers)}"
            )
        self._aux_hidden_state_layers = expected

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.DFLASH_AUX_HIDDEN_STATE_LAYERS

    def _allocate_compiled_outputs(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return {
            name: torch.empty(
                (num_tokens, self.config.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            for name in ("hidden_states", *self._dflash_output_names)
        }

    def _format_compiled_outputs(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = outputs["hidden_states"]
        if not self._aux_hidden_state_layers:
            return hidden_states
        return hidden_states, [outputs[name] for name in self._dflash_output_names]

    def _format_profile_output(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if not self._aux_hidden_state_layers:
            return inputs_embeds
        return inputs_embeds, [
            torch.empty_like(inputs_embeds) for _ in self._dflash_output_names
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = OrnithMxfp4StreamingLoader(self.manifest, max_pending_layers=2)
        loaded: set[str] = set()
        shell = {
            "model.language_model.embed_tokens.weight": self.embed_tokens,
            "lm_head.weight": self.lm_head,
        }
        device = torch.device("cuda", torch.cuda.current_device())
        for name, tensor in weights:
            module = shell.get(name)
            if module is not None:
                module.weight_loader(module.weight, tensor)
                loaded.add(name)
                continue
            transformed = loader.consume(name, tensor)
            if transformed is None:
                raise ValueError(f"unsupported Ornith checkpoint tensor {name}")
            if not (
                name.startswith("model.visual.")
                or name.startswith("mtp.")
                or name.startswith("model.language_model.layers.")
                and int(name.split(".")[3]) >= loader.num_layers
            ):
                loaded.add(name)
            for target_name, value in transformed:
                self.compiled_model.set_constant_with_tensor(
                    target_name, value.to(device=device).contiguous()
                )
        missing_shell = sorted(set(shell) - loaded)
        if missing_shell:
            raise ValueError(
                "Ornith checkpoint is missing runtime shell tensors: "
                + ", ".join(missing_shell)
            )
        loader.finish()
        if loader.emitted_target_names != loader.expected_target_names:
            raise ValueError("Ornith loader did not bind every compiled constant")
        return loaded
