"""Cached per-preset AdaLN modulation tables for the pruned H3 Ref2VA runtime.

The runtime table contract is produced by the quantizer's ``precompute_adaln.py``:

    blocks.<n>.adaln_t_table_<preset>   bf16 [T*3, 6*5376]
    final_layer.adaln_t_table_<preset>  bf16 [T, 2*5376]
    adaln_t_values_<preset>             f32  [T]

Rows are ``global_timestep_index*3 + token_tag`` with tags video=0, text=1,
audio=2 and payload order shift_msa, scale_msa, gate_msa, shift_mlp,
scale_mlp, gate_mlp. The global index is the position in
``adaln_t_values_<preset>``, not the per-step unique list.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from safetensors import safe_open


FORMAT = "h3_adaln_per_modality_cached"
ABI_VERSION = 1
HIDDEN = 5376
BLOCKS = 50
PARAMS = 6
MODALITIES = 3
ROW_ORDER = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
TOKEN_TAGS = {"video": 0, "text": 1, "audio": 2}
PRESETS = {
    "base50": {"steps": 50, "shift_video": 12.0, "shift_audio": 3.0},
    "turbo4": {"steps": 4, "shift_video": 6.0, "shift_audio": 3.0},
    "turbo8": {"steps": 8, "shift_video": 6.0, "shift_audio": 3.0},
}


@dataclass(frozen=True)
class AdalnTables:
    preset: str
    steps: int
    shift_video: float
    shift_audio: float
    hidden: int
    values: torch.Tensor
    blocks: tuple
    final: torch.Tensor

    def to(self, device):
        device = torch.device(device)
        if self.values.device == device:
            return self
        return AdalnTables(self.preset, self.steps, self.shift_video, self.shift_audio,
                           self.hidden, self.values.to(device),
                           tuple(table.to(device) for table in self.blocks),
                           self.final.to(device))


@dataclass(frozen=True)
class PrunedAdaln:
    table: torch.Tensor
    blocks: tuple
    final: tuple

    def to(self, device):
        device = torch.device(device)
        if self.table.device == device:
            return self
        return PrunedAdaln(self.table.to(device),
                            tuple((weight.to(device), bias.to(device))
                                  for weight, bias in self.blocks),
                            (self.final[0].to(device), self.final[1].to(device)))


class CachedAdalnBlock(torch.nn.Module):
    """Drop-in replacement for ``DiTBlock.adaln_proj(t_emb)``.

    ComfyUI indexes ``mod_segments`` with the per-step local timestep index.
    ``t_emb`` carries the global preset index from ``_GlobalIndexEmbedder``,
    so remap the local ``t*3+tag`` rows before chunking.
    """

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table, persistent=False)

    def forward(self, t_emb):
        index = t_emb.reshape(-1).to(torch.long)
        tags = torch.arange(MODALITIES, device=index.device)
        rows = (index[:, None]*MODALITIES + tags[None, :]).reshape(-1)
        return tuple(self.table.index_select(0, rows).chunk(PARAMS, dim=-1))


class CachedAdalnOut(torch.nn.Module):
    """Drop-in replacement for ``FinalLayer.adaln_proj(t_emb)``."""

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table, persistent=False)

    def forward(self, t_emb):
        index = t_emb.reshape(-1).to(torch.long)
        return tuple(self.table.index_select(0, index).chunk(2, dim=-1))


class _GlobalIndexEmbedder(torch.nn.Module):
    """Return each unique timestep's global row index for the cached tables."""

    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", values, persistent=False)

    def forward(self, t):
        values = self.values.to(t.device)
        flat = t.reshape(-1).to(torch.float32)
        index = torch.searchsorted(values, flat).clamp(max=values.numel()-1)
        if not bool(torch.isclose(values[index], flat, atol=1e-6, rtol=0).all()):
            raise ValueError("H3 timestep is missing from the cached AdaLN table")
        return index.to(torch.float32).unsqueeze(1)


def load_pruned_adaln(path, prefix="", device="cpu"):
    """Load the starsfriday 8-dim AdaLN table and per-block weights."""
    reader = _TensorReader(path)
    try:
        table = reader.get(prefix+"adaln_t_table").to(torch.float32)
        blocks = tuple((reader.get(f"{prefix}blocks.{index}.adaln_proj.linear.weight"),
                        reader.get(f"{prefix}blocks.{index}.adaln_proj.linear.bias"))
                       for index in range(BLOCKS))
        final = (reader.get(prefix+"final_layer.adaln_proj.linear.weight"),
                 reader.get(prefix+"final_layer.adaln_proj.linear.bias"))
    finally:
        reader.close()
    if tuple(table.shape) != (1025, 8):
        raise ValueError("H3 pruned checkpoint is missing adaln_t_table [1025, 8]")
    for index, (weight, bias) in enumerate(blocks):
        if tuple(weight.shape) != (PARAMS*HIDDEN*MODALITIES, 8) or tuple(bias.shape) != (PARAMS*HIDDEN*MODALITIES,):
            raise ValueError(f"H3 pruned blocks.{index} AdaLN layout mismatch")
    if tuple(final[0].shape) != (2*HIDDEN, 8) or tuple(final[1].shape) != (2*HIDDEN,):
        raise ValueError("H3 pruned final_layer AdaLN layout mismatch")
    return PrunedAdaln(table, blocks, final).to(device)


def install_pruned(model, pruned):
    """Install the 8-dim pruned table/weights into a stock H3 model."""
    if len(model.blocks) != len(pruned.blocks):
        raise ValueError("H3 pruned AdaLN block count mismatch")
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = pruned.table.device
    pruned = pruned.to(device)
    if hasattr(model, "adaln_t_table"):
        if tuple(model.adaln_t_table.shape) != tuple(pruned.table.shape):
            raise ValueError("H3 model AdaLN table shape does not match the checkpoint")
        model.adaln_t_table = pruned.table
    else:
        model.register_buffer("adaln_t_table", pruned.table)
    model.use_adaln_curves = True
    for block, (weight, bias) in zip(model.blocks, pruned.blocks):
        linear = block.adaln_proj.linear
        if tuple(linear.weight.shape) != tuple(weight.shape) or tuple(linear.bias.shape) != tuple(bias.shape):
            raise ValueError("H3 model AdaLN projection does not match the checkpoint")
        linear.weight.data.copy_(weight.to(device=device, dtype=linear.weight.dtype))
        linear.bias.data.copy_(bias.to(device=device, dtype=linear.bias.dtype))
    linear = model.final_layer.adaln_proj.linear
    if tuple(linear.weight.shape) != tuple(pruned.final[0].shape):
        raise ValueError("H3 model final AdaLN projection does not match the checkpoint")
    linear.weight.data.copy_(pruned.final[0].to(device=device, dtype=linear.weight.dtype))
    linear.bias.data.copy_(pruned.final[1].to(device=device, dtype=linear.bias.dtype))
    return pruned


def check_pruned_adaln_layout(shapes, prefix=""):
    """Validate the starsfriday 8-dim pruned AdaLN checkpoint layout.

    ``shapes`` maps state-dict keys to shapes; no tensors are loaded.
    """
    def shape(name):
        value = shapes.get(prefix+name)
        return None if value is None else tuple(int(item) for item in value)

    if shape("adaln_t_table") != (1025, 8):
        raise ValueError("H3 pruned checkpoint is missing adaln_t_table [1025, 8]")
    if prefix+"time_embedder.proj_in.weight" in shapes:
        raise ValueError("H3 checkpoint has a full time embedder, not the pruned table")
    for index in range(BLOCKS):
        weight = shape(f"blocks.{index}.adaln_proj.linear.weight")
        bias = shape(f"blocks.{index}.adaln_proj.linear.bias")
        if weight != (PARAMS*HIDDEN*MODALITIES, 8) or bias != (PARAMS*HIDDEN*MODALITIES,):
            raise ValueError(f"H3 pruned blocks.{index} AdaLN layout mismatch")
    if (shape("final_layer.adaln_proj.linear.weight") != (2*HIDDEN, 8)
            or shape("final_layer.adaln_proj.linear.bias") != (2*HIDDEN,)):
        raise ValueError("H3 pruned final_layer AdaLN layout mismatch")
    return True


def _manifest_spec(path):
    path = Path(path)
    names = ("adaln_manifest.json", "manifest.json")
    roots = (path,) if path.is_dir() else (path.parent,)
    for root in roots:
        for name in names:
            candidate = root/name
            if not candidate.exists():
                continue
            data = json.loads(candidate.read_text())
            if isinstance(data, dict):
                for key in ("h3_adaln_tables", "adaln_tables"):
                    if key in data:
                        return data[key]
            if name == "adaln_manifest.json":
                return data
    return None


class _TensorReader:
    def __init__(self, path):
        self.path = Path(path)
        self._single = None
        self._handles = {}
        self._weight_map = None
        if self.path.is_dir():
            sidecar = self.path/"adaln_tables.safetensors"
            if sidecar.exists():
                self._single = safe_open(str(sidecar), framework="pt", device="cpu")
            else:
                index = self.path/"model.safetensors.index.json"
                if not index.exists():
                    raise FileNotFoundError(f"No AdaLN tables or checkpoint index in {self.path}")
                self._weight_map = json.loads(index.read_text())["weight_map"]
        elif self.path.suffix == ".safetensors":
            self._single = safe_open(str(self.path), framework="pt", device="cpu")
        else:
            raise FileNotFoundError(f"Not an AdaLN table or checkpoint path: {self.path}")

    def get(self, key):
        if self._single is not None:
            try:
                return self._single.get_tensor(key)
            except KeyError as error:
                raise ValueError(f"Missing AdaLN tensor: {key}") from error
        shard = self._weight_map.get(key)
        if shard is None:
            raise ValueError(f"Missing AdaLN tensor: {key}")
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.path/shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle.get_tensor(key)

    def close(self):
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        if self._single is not None:
            self._single.__exit__(None, None, None)


def _names(spec, preset):
    if spec is not None:
        entry = (spec.get("presets") or {}).get(preset)
        if entry is None:
            raise ValueError(f"AdaLN manifest has no preset {preset}")
        if (isinstance(entry.get("values"), str) and isinstance(entry.get("blocks"), str)
                and isinstance(entry.get("final"), str)):
            return entry["values"], entry["blocks"], entry["final"]
    return (f"adaln_t_values_{preset}", f"blocks.{{block}}.adaln_t_table_{preset}",
            f"final_layer.adaln_t_table_{preset}")


def _manifest_values(spec, preset):
    if spec is None:
        return None
    entry = (spec.get("presets") or {}).get(preset)
    if entry is None or not isinstance(entry.get("values"), list):
        return None
    return torch.tensor(entry["values"], dtype=torch.float32)


def _validate_manifest(spec, preset):
    if spec is None:
        return
    if "format" in spec:
        if spec.get("format") != FORMAT or int(spec.get("abi_version", -1)) != ABI_VERSION:
            raise ValueError("Unsupported H3 AdaLN table format/ABI")
        if spec.get("row_index") != "timestep_index*3+token_tag":
            raise ValueError("Unsupported H3 AdaLN row index")
        if tuple(spec.get("row_order") or ()) != ROW_ORDER:
            raise ValueError("Unsupported H3 AdaLN row order")
        if dict(spec.get("token_tags") or {}) != TOKEN_TAGS:
            raise ValueError("Unsupported H3 AdaLN token tags")
        entry = (spec.get("presets") or {}).get(preset)
        if entry is None:
            raise ValueError(f"AdaLN manifest has no preset {preset}")
        expected = PRESETS[preset]
        if int(entry.get("steps", -1)) != expected["steps"]:
            raise ValueError(f"H3 AdaLN {preset} steps mismatch")
        if float(entry.get("shift_video", -1.0)) != expected["shift_video"]:
            raise ValueError(f"H3 AdaLN {preset} video shift mismatch")
        if float(entry.get("shift_audio", -1.0)) != expected["shift_audio"]:
            raise ValueError(f"H3 AdaLN {preset} audio shift mismatch")
        return
    if spec.get("index") != "timestep_index*3+token_tag":
        raise ValueError("Unsupported H3 AdaLN row index")
    if tuple(spec.get("row_order") or ()) != ROW_ORDER:
        raise ValueError("Unsupported H3 AdaLN row order")
    if dict(spec.get("token_tags") or {}) != TOKEN_TAGS:
        raise ValueError("Unsupported H3 AdaLN token tags")
    row_width = int(spec.get("row_width", -1))
    if row_width <= 0 or row_width % PARAMS:
        raise ValueError("Unsupported H3 AdaLN row width")
    if int(spec.get("final_row_width", -1)) != 2*(row_width//PARAMS):
        raise ValueError("Unsupported H3 AdaLN final row width")
    entry = (spec.get("presets") or {}).get(preset)
    if entry is None:
        raise ValueError(f"AdaLN manifest has no preset {preset}")
    values = entry.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError(f"H3 AdaLN {preset} values are missing")
    if int(entry.get("num_timesteps", len(values))) != len(values):
        raise ValueError(f"H3 AdaLN {preset} timestep count mismatch")


def _validate(values, blocks, final, preset, hidden):
    if values.ndim != 1 or values.numel() < 2 or not bool(torch.isfinite(values).all()):
        raise ValueError("H3 AdaLN values must be a finite 1-D array")
    if bool((values[1:] < values[:-1]).any()):
        raise ValueError("H3 AdaLN values must be sorted ascending")
    for pinned in (0.999, 1.0):
        if not bool(torch.isclose(values, torch.tensor(pinned), atol=1e-6, rtol=0).any()):
            raise ValueError(f"H3 AdaLN values are missing pinned timestep {pinned}")
    rows = values.numel()
    if len(blocks) != BLOCKS:
        raise ValueError(f"H3 AdaLN needs {BLOCKS} block tables")
    for index, table in enumerate(blocks):
        if tuple(table.shape) != (rows*MODALITIES, PARAMS*hidden):
            raise ValueError(f"H3 AdaLN blocks.{index} has shape {tuple(table.shape)}")
        if table.dtype != torch.bfloat16:
            raise ValueError(f"H3 AdaLN blocks.{index} must be bfloat16")
    if tuple(final.shape) != (rows, 2*hidden):
        raise ValueError(f"H3 AdaLN final table has shape {tuple(final.shape)}")
    if final.dtype != torch.bfloat16:
        raise ValueError("H3 AdaLN final table must be bfloat16")
    expected = PRESETS[preset]
    return AdalnTables(preset, expected["steps"], expected["shift_video"],
                       expected["shift_audio"], hidden, values, tuple(blocks), final)


def prepare_state_dict(state_dict):
    """Add tiny AdaLN placeholders so stock H3 detection/load accepts a cached checkpoint.

    The pruned checkpoint drops ``time_embedder`` and every ``adaln_proj``.
    ComfyUI's H3 detector still needs one of those keys to build the model.
    The one-dimension curve placeholders make detection choose the curve path;
    ``install()`` replaces them with the real cached tables before inference.
    """
    keys = set(state_dict)
    suffix = "video_patch_proj.weight"
    prefix = next((key[:-len(suffix)] for key in keys if key.endswith(suffix)), None)
    if prefix is None or f"{prefix}adaln_t_values_base50" not in keys:
        return state_dict
    if f"{prefix}adaln_t_table" in keys or f"{prefix}time_embedder.proj_in.weight" in keys:
        return state_dict
    prepared = dict(state_dict)
    prepared[f"{prefix}adaln_t_table"] = torch.zeros((1, 1), dtype=torch.float32)
    for index in range(BLOCKS):
        prepared[f"{prefix}blocks.{index}.adaln_proj.linear.weight"] = torch.zeros(
            (PARAMS*HIDDEN*MODALITIES, 1), dtype=torch.float32)
        prepared[f"{prefix}blocks.{index}.adaln_proj.linear.bias"] = torch.zeros(
            (PARAMS*HIDDEN*MODALITIES,), dtype=torch.float32)
    prepared[f"{prefix}final_layer.adaln_proj.linear.weight"] = torch.zeros(
        (2*HIDDEN, 1), dtype=torch.float32)
    prepared[f"{prefix}final_layer.adaln_proj.linear.bias"] = torch.zeros(
        (2*HIDDEN,), dtype=torch.float32)
    return prepared


def install_checkpoint_loader():
    """Wrap ComfyUI's diffusion loader so cached checkpoints get the placeholders."""
    import comfy.sd
    loader = getattr(comfy.sd, "load_diffusion_model_state_dict", None)
    if loader is None or getattr(loader, "_paiton_adaln", False):
        return
    def load(state_dict, *args, **kwargs):
        return loader(prepare_state_dict(state_dict), *args, **kwargs)
    load._paiton_adaln = True
    comfy.sd.load_diffusion_model_state_dict = load


def load_tables(path, preset, device="cpu"):
    if preset not in PRESETS:
        raise ValueError(f"Unknown H3 AdaLN preset: {preset}")
    spec = _manifest_spec(path)
    _validate_manifest(spec, preset)
    if spec is None:
        hidden = HIDDEN
    elif "format" in spec:
        hidden = int(spec.get("hidden", HIDDEN))
    else:
        hidden = int(spec.get("row_width", PARAMS*HIDDEN))//PARAMS
    manifest_values = _manifest_values(spec, preset)
    values_name, blocks_name, final_name = _names(spec, preset)
    reader = _TensorReader(path)
    try:
        try:
            values = reader.get(values_name).to(torch.float32).reshape(-1)
        except ValueError:
            if manifest_values is None:
                raise
            values = manifest_values
        blocks = tuple(reader.get(blocks_name.format(block=index)) for index in range(BLOCKS))
        final = reader.get(final_name)
    finally:
        reader.close()
    if manifest_values is not None and not torch.equal(values, manifest_values):
        raise ValueError("H3 AdaLN values tensor does not match the manifest")
    tables = _validate(values, blocks, final, preset, hidden)
    return tables.to(device)


def check_layout(layout, tables):
    if not torch.equal(layout.timestep_values.cpu(), tables.values.cpu()):
        raise ValueError("H3 layout timestep values do not match the cached AdaLN table")
    if layout.adaln_indices.numel() and int(layout.adaln_indices.max()) >= tables.values.numel()*MODALITIES:
        raise ValueError("H3 layout AdaLN index exceeds the cached table")
    return layout.adaln_indices


def install(model, tables):
    if len(model.blocks) != BLOCKS:
        raise ValueError(f"H3 AdaLN needs {BLOCKS} blocks")
    if int(getattr(model, "hidden_size", tables.hidden)) != tables.hidden:
        raise ValueError("H3 AdaLN table hidden size does not match the model")
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = tables.values.device
    tables = tables.to(device)
    for index, block in enumerate(model.blocks):
        block.adaln_proj = CachedAdalnBlock(tables.blocks[index])
    model.final_layer.adaln_proj = CachedAdalnOut(tables.final)
    model.use_adaln_curves = False
    model.time_embedder = _GlobalIndexEmbedder(tables.values)
    model._paiton_adaln_tables = tables
    return tables
