# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import glob
import os
import time
from collections.abc import Generator, Iterable
from typing import cast

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.torchao import torchao_version_at_least
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    get_quant_config,
    instanttensor_weights_iterator,
    maybe_download_from_modelscope,
    multi_thread_pt_weights_iterator,
    multi_thread_safetensors_weights_iterator,
    np_cache_weights_iterator,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.tracing import instrument
from vllm.transformers_utils.repo_utils import list_filtered_repo_files


# --- paiton (start-time item A1): skip reading target tensors the W3 policy replaces. The W3 adapter (loaded by its
# .pth hook when PAITON_W3_DECODE=1) exposes pre_read_skip(language_model_only) -> predicate on checkpoint tensor names,
# or None when W3 is off. Skipped tensors are never materialised from the safetensors files; their parameters are
# zero-filled after load and counted as loaded (the adapter's build replaces them from the calibration output). The
# predicate refuses to start when the calibration output does not cover a module it would skip.
def _paiton_pre_read_skip(model_config: ModelConfig):
    import sys

    adapter = sys.modules.get("paiton_w3_decode_adapter")
    factory = getattr(adapter, "pre_read_skip", None)
    if factory is None:
        return None
    # ModelConfig.language_model_only is an InitVar; the value lives on multimodal_config (None for text-only models).
    mm = getattr(model_config, "multimodal_config", None)
    language_model_only = bool(getattr(mm, "language_model_only", False)) or bool(getattr(model_config, "language_model_only", False))
    return factory(language_model_only)


def _paiton_filtered_safetensors_iterator(
    hf_weights_files: list[str], use_tqdm_on_load: bool, skip, skipped: list[str]
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """The lazy safetensors iterator with a name filter: skipped names are decided up front over the headers (so an
    uncovered module fails before the first byte is read) and never fetched."""
    from safetensors import safe_open
    from tqdm.auto import tqdm

    from vllm.model_executor.model_loader.weight_utils import _BAR_FORMAT, _natural_sort_key, enable_tqdm

    import json
    import struct

    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)
    plan: dict[str, list[str]] = {}
    ranges: dict[str, dict[str, tuple[int, int]]] = {}
    for st_file in sorted_files:
        with open(st_file, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        base = 8 + header_len
        keep = []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if skip(name):
                skipped.append(name)
            else:
                keep.append(name)
                a, b = meta["data_offsets"]
                ranges.setdefault(st_file, {})[name] = (base + a, b - a)
        plan[st_file] = keep
    logger.info(
        "[paiton.w3] pre-read skip: %d tensors not read (W3-replaced linears, MTP, vision); %d kept",
        len(skipped), sum(len(v) for v in plan.values()),
    )
    # A6 (opt-in, PAITON_LOADER_DROP_PAGECACHE=1): once a tensor has been handed on (its copy to the device happens
    # before the next one is requested), advise the kernel that its file range is not needed again, so the weight
    # stream stops evicting the Python files the next start imports. Advisory; bytes are unchanged.
    drop = os.environ.get("PAITON_LOADER_DROP_PAGECACHE", "0") == "1"
    for st_file in tqdm(
        sorted_files, desc="Loading safetensors checkpoint shards (W3 pre-read skip)",
        disable=not enable_tqdm(use_tqdm_on_load), bar_format=_BAR_FORMAT,
    ):
        fd = os.open(st_file, os.O_RDONLY) if drop else None
        previous = None
        try:
            with safe_open(st_file, framework="pt") as f:
                for name in plan[st_file]:
                    tensor = f.get_tensor(name)
                    if fd is not None and previous is not None:
                        off, length = ranges[st_file][previous]
                        os.posix_fadvise(fd, off, length, os.POSIX_FADV_DONTNEED)
                    previous = name
                    yield name, tensor
        finally:
            if fd is not None:
                if previous is not None:
                    off, length = ranges[st_file][previous]
                    os.posix_fadvise(fd, off, length, os.POSIX_FADV_DONTNEED)
                os.close(fd)


def _paiton_skipped_module_prefixes(skipped: list[str]) -> set[tuple[int, str]]:
    """(layer, served module) pairs whose checkpoint tensors were skipped, from the adapter's policy."""
    import re
    import sys

    policy = sys.modules.get("paiton_w3_decode_policy")
    if policy is None:
        return set()
    part_to_module = {part: d["module"] for d in policy.SITES.values() for part, _ in d["parts"]}
    out = set()
    rx = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+?)\.(weight\w*|input_global_scale|bias)$")
    for name in skipped:
        m = rx.match(name)
        if m and m.group(2) in part_to_module:
            out.add((int(m.group(1)), part_to_module[m.group(2)]))
    return out


def _paiton_zero_fill_skipped(model: nn.Module, skipped: list[str], loaded_weights) -> int:
    """Zero the parameters of every skipped (layer, module) and count them as loaded."""
    import re

    targets = _paiton_skipped_module_prefixes(skipped)
    if not targets:
        return 0
    rx = re.compile(r"layers\.(\d+)\.(.+)$")
    n = 0
    with torch.no_grad():
        for mod_name, module in model.named_modules():
            m = rx.search(mod_name)
            if not m or (int(m.group(1)), m.group(2)) not in targets:
                continue
            for pname, param in module.named_parameters(recurse=False):
                param.data.zero_()
                if loaded_weights is not None:
                    loaded_weights.add(f"{mod_name}.{pname}")
                n += 1
    logger.info("[paiton.w3] pre-read skip: %d parameters of %d modules zero-filled until the W3 build replaces them",
                n, len(targets))
    return n

logger = init_logger(__name__)


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: str | None
        """The optional model revision."""

        subfolder: str | None = None
        """The subfolder inside the model repo."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    _paiton_skip = None
    _paiton_skipped: list[str] = []

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self.local_expert_ids: set[int] | None = None

        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError(
                f"model_loader_extra_config must be a dict for load format "
                f"{load_config.load_format}, got {type(extra_config).__name__}"
            )
        allowed_keys = {
            "enable_multithread_load",
            "num_threads",
            "enable_weights_track",
        }
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

        enable_multithread_load = extra_config.get("enable_multithread_load", False)
        if not isinstance(enable_multithread_load, bool):
            raise ValueError(
                f"enable_multithread_load must be a bool, got "
                f"{type(enable_multithread_load).__name__}"
            )
        num_threads = extra_config.get("num_threads")
        if num_threads is not None and not (
            isinstance(num_threads, int) and num_threads > 0
        ):
            raise ValueError(
                f"num_threads must be a positive integer, got {num_threads!r}"
            )

        self.enable_weights_track: bool | None = extra_config.get(
            "enable_weights_track", None
        )

        # The multi-thread loader ignores safetensors_load_strategy, so reject
        # the combination instead of silently dropping the requested strategy.
        if extra_config.get("enable_multithread_load") and (
            load_config.safetensors_load_strategy not in (None, "lazy")
        ):
            raise ValueError(
                "enable_multithread_load does not support "
                "safetensors_load_strategy="
                f"{load_config.safetensors_load_strategy!r}; the multi-thread "
                "loader only implements the default lazy strategy."
            )

    def _prepare_weights(
        self,
        model_name_or_path: str,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = (
            maybe_download_from_modelscope(model_name_or_path, revision)
            or model_name_or_path
        )

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME

        # First check for 'auto' format that mistral files format are present.
        # This is to load mistral models with official format by default.
        if load_format == "auto":
            load_format = (
                "mistral"
                if len(
                    list_filtered_repo_files(
                        model_name_or_path=model_name_or_path,
                        allow_patterns=["consolidated*.safetensors"],
                        revision=revision,
                    )
                )
                > 0
                else "hf"
            )

        # Some quantized models use .pt files for storing the weights.
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        elif (
            load_format == "safetensors"
            or load_format == "fastsafetensors"
            or load_format == "instanttensor"
        ):
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == "mistral":
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == "pt":
            allow_patterns = ["*.pt"]
        elif load_format == "npcache":
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        # Don't fall back to .pt for explicit safetensors formats; otherwise a
        # .pt file is matched and later opened as safetensors.
        if fall_back_to_pt and not use_safetensors:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                subfolder=subfolder,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        if subfolder is not None:
            hf_folder = os.path.join(hf_folder, subfolder)

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern.endswith(".safetensors"):
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local and len(hf_weights_files) > 1:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    cache_dir=self.load_config.download_dir,
                    subfolder=subfolder,
                    revision=revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        extra_config = self.load_config.model_loader_extra_config
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if self.load_config.load_format == "npcache":
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        elif use_safetensors:
            if self.load_config.load_format == "fastsafetensors":
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            elif self.load_config.load_format == "instanttensor":
                weights_iterator = instanttensor_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            else:
                if extra_config.get("enable_multithread_load"):
                    weights_iterator = multi_thread_safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        max_workers=extra_config.get(
                            "num_threads", self.DEFAULT_NUM_THREADS
                        ),
                    )
                elif (
                    self._paiton_skip is not None
                    and self.load_config.safetensors_load_strategy in (None, "lazy")
                    and self.local_expert_ids is None
                ):
                    weights_iterator = _paiton_filtered_safetensors_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        self._paiton_skip,
                        self._paiton_skipped,
                    )
                else:
                    weights_iterator = safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        self.load_config.safetensors_load_strategy,
                        local_expert_ids=self.local_expert_ids,
                        safetensors_prefetch_num_threads=(
                            self.load_config.safetensors_prefetch_num_threads
                        ),
                        safetensors_prefetch_block_size=(
                            self.load_config.safetensors_prefetch_block_size
                        ),
                    )
        else:
            if extra_config.get("enable_multithread_load"):
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                )
            else:
                weights_iterator = pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_name_or_path=model_config.model,
            subfolder=None,
            revision=model_config.revision,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )

    def _init_ep_weight_filter(self, model_config: ModelConfig) -> None:
        """Compute local expert ids for EP weight filtering.

        When expert parallelism is active, each rank only needs a subset of
        expert weights.  By computing the set upfront we can skip non-local
        expert tensors *before* reading them from disk.
        """
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config

        if not (
            model_config.is_moe
            and parallel_config.enable_expert_parallel
            and parallel_config.enable_ep_weight_filter
        ):
            return

        # When EPLB is enabled, redundant physical expert slots may map to
        # logical experts that belong to other ranks in the default partition.
        # The weight loader needs to see ALL logical expert weights so it can
        # populate these redundant slots.  Skip the filter entirely.
        if parallel_config.enable_eplb:
            return

        num_experts = model_config.get_num_experts()
        if num_experts <= 0:
            return

        # EP size/rank computation mirrors FusedMoEParallelConfig.make():
        #   ep_size = dp_size * pcp_size * tp_size (flattened)
        #   ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
        from vllm.distributed import (
            get_dp_group,
            get_pcp_group,
            get_tensor_model_parallel_rank,
        )

        dp_size = parallel_config.data_parallel_size
        tp_size = parallel_config.tensor_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        ep_size = dp_size * pcp_size * tp_size
        ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

        self.local_expert_ids = compute_local_expert_ids(
            num_experts,
            ep_size,
            ep_rank,
            placement=parallel_config.expert_placement_strategy,
        )
        if self.local_expert_ids is not None:
            logger.info_once(
                "EP weight filter: ep_size=%d, ep_rank=%d, loading %d/%d experts",
                ep_size,
                ep_rank,
                len(self.local_expert_ids),
                num_experts,
            )

    @instrument(span_name="Load weights")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        if model_config.quantization == "torchao":
            quant_config = get_quant_config(model_config, self.load_config)
            if (
                hasattr(quant_config, "is_checkpoint_torchao_serialized")
                and quant_config.is_checkpoint_torchao_serialized
                and torchao_version_at_least("0.15.0")
            ):
                self.load_config.safetensors_load_strategy = "torchao"

        self._init_ep_weight_filter(model_config)
        # --- paiton A1: decide before the first byte is read which target tensors the W3 build makes redundant
        self._paiton_skip = _paiton_pre_read_skip(model_config)
        self._paiton_skipped = []

        # --- radiance (patch_tp3_pad.py): pad checkpoint tensors with dummy heads before the
        # sharding weight loaders see them. Returns the iterator untouched unless this model's
        # config was padded (RADIANCE_TP_PAD).
        _weights = self.get_all_weights(model_config, model)
        try:
            import radiance_tp3pad as _radiance_tp3pad
        except ImportError:
            if os.environ.get("RADIANCE_TP_PAD", "").strip() not in ("", "0"):
                raise
        else:
            _weights = _radiance_tp3pad.pad_weights(_weights, model_config)
        loaded_weights = model.load_weights(_weights)
        if self._paiton_skipped:
            if loaded_weights is None:
                loaded_weights = set()
            _paiton_zero_fill_skipped(model, self._paiton_skipped, loaded_weights)

        self.counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking by default.
        default_enable_weights_track = (
            model_config.quantization is None and loaded_weights is not None
        )
        enable_weights_track = (
            self.enable_weights_track
            if self.enable_weights_track is not None
            else default_enable_weights_track
        )
        if enable_weights_track:
            self.track_weights_loading(model, loaded_weights)

    def track_weights_loading(
        self, model: nn.Module, loaded_weights: set[str] | None
    ) -> None:
        weights_to_load = {name for name, _ in model.named_parameters()}
        if loaded_weights is not None:
            # ignore online quantization scales
            for name, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                has_online_quant = getattr(quant_method, "uses_meta_device", False)
                has_postprocess_quant = getattr(
                    quant_method, "process_weights_after_loading", None
                )
                # ignore kv_cache scale and online quant scale,
                # which can be missing in checkpoints
                if has_online_quant or has_postprocess_quant:
                    for param_name, _ in module.named_parameters():
                        full_name = f"{name}.{param_name}" if name else param_name
                        loaded_weights.add(full_name)
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )
