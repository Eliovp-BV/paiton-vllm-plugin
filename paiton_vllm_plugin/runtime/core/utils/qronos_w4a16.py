"""Strict Quark Qronos/AWQ checkpoint-to-kernel W4A16 transformation.

Both formats store packed signed INT4 weights as I32 [K, N/8] using Quark's
reorder convention and packed I32 zero-point metadata as [K/128, N/8].
Qronos checkpoint scales are F32; AWQ checkpoint scales are BF16. Paiton's
gfx12 kernel layout is ExLlama-shuffled I32 [N_padded, K/8]. The established
artifact ABI uses F32 kernel scales; the explicitly versioned Qwen3.8
performance ABI casts one Qronos scale tensor at a time to BF16 while
transposing it to [N_padded, K/128].

No BF16 weight expansion is created. Repacking uses bounded output chunks so
large projections do not also allocate a full unpacked I32 matrix.
"""

from dataclasses import dataclass

import torch


QUARK_REORDER = (0, 4, 1, 5, 2, 6, 3, 7)
EXLLAMA_K_SHIFTS = (0, 16, 4, 20, 8, 24, 12, 28)
GROUP_SIZE = 128
LAYOUT_VERSION = "paiton_w4a16_g128_v1"


@dataclass(frozen=True)
class QronosW4A16Weights:
    packed_weight: torch.Tensor
    scales: torch.Tensor
    input_size: int
    output_size: int
    padded_output_size: int
    group_size: int = GROUP_SIZE
    layout_version: str = LAYOUT_VERSION


@dataclass(frozen=True)
class QronosW3A16DecodeWeights:
    packed_weight: torch.Tensor
    scales: torch.Tensor


def _require_cpu_contiguous(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be transformed on CPU, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_checkpoint_tensors(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    packed_zero_points: torch.Tensor,
    group_size: int,
    *,
    expected_scale_dtype: torch.dtype,
    format_name: str,
) -> tuple[int, int]:
    if group_size != GROUP_SIZE:
        raise ValueError(f"{format_name} W4A16 requires group_size=128, got {group_size}")
    for tensor, name in (
        (packed_weight, "packed_weight"),
        (scales, "scales"),
        (packed_zero_points, "packed_zero_points"),
    ):
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be rank 2, got shape {tuple(tensor.shape)}")
        _require_cpu_contiguous(tensor, name)
    if packed_weight.dtype is not torch.int32:
        raise ValueError(f"packed_weight must have dtype int32, got {packed_weight.dtype}")
    if scales.dtype is not expected_scale_dtype:
        expected_scale_name = str(expected_scale_dtype).removeprefix("torch.")
        raise ValueError(
            f"{format_name} scales must have dtype {expected_scale_name}, "
            f"got {scales.dtype}"
        )
    if packed_zero_points.dtype is not torch.int32:
        raise ValueError(
            f"packed_zero_points must have dtype int32, got {packed_zero_points.dtype}"
        )

    input_size, packed_output_size = packed_weight.shape
    output_size = scales.shape[1]
    if input_size % group_size != 0:
        raise ValueError(
            f"input size K={input_size} must be divisible by group_size={group_size}"
        )
    if input_size % 8 != 0:
        raise ValueError(f"input size K={input_size} must be divisible by pack factor 8")
    if output_size <= 0 or output_size > packed_output_size * 8:
        raise ValueError(
            f"scale output size N={output_size} is incompatible with packed capacity "
            f"{packed_output_size * 8}"
        )
    if packed_output_size != (output_size + 7) // 8:
        raise ValueError(
            f"packed output dimension must be ceil(N/8)={(output_size + 7) // 8}, "
            f"got {packed_output_size}"
        )
    expected_scale_shape = (input_size // group_size, output_size)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"scales shape must be {expected_scale_shape}, got {tuple(scales.shape)}"
        )
    expected_zero_shape = (input_size // group_size, packed_output_size)
    if tuple(packed_zero_points.shape) != expected_zero_shape:
        raise ValueError(
            "packed_zero_points shape must be "
            f"{expected_zero_shape}, got {tuple(packed_zero_points.shape)}"
        )
    if torch.count_nonzero(packed_zero_points).item() != 0:
        raise ValueError(
            f"symmetric {format_name} checkpoint zero-point metadata must contain only zero"
        )
    if not torch.isfinite(scales).all().item():
        raise ValueError(f"{format_name} scales must be finite")
    if (scales < 0).any().item():
        raise ValueError(f"{format_name} scales must be non-negative")
    return input_size, output_size


def repack_qronos_reorder_to_exllama(
    packed_weight: torch.Tensor,
    output_size: int,
    padded_output_size: int,
    *,
    output_chunk_size: int = 256,
) -> torch.Tensor:
    """Repack signed Quark output-packed I4 into biased ExLlama K-packing."""
    if output_chunk_size <= 0 or output_chunk_size % 8 != 0:
        raise ValueError("output_chunk_size must be a positive multiple of 8")
    if padded_output_size < output_size or padded_output_size % 8 != 0:
        raise ValueError(
            "padded_output_size must be at least output_size and divisible by 8"
        )

    input_size = packed_weight.shape[0]
    packed_kernel_weight = torch.zeros(
        (padded_output_size, input_size // 8), dtype=torch.int32, device="cpu"
    )
    source_shifts = tuple(index * 4 for index in QUARK_REORDER)

    for output_start in range(0, output_size, output_chunk_size):
        output_end = min(output_start + output_chunk_size, output_size)
        source_word_start = output_start // 8
        source_word_end = (output_end + 7) // 8
        source = packed_weight[:, source_word_start:source_word_end]

        # Quark symmetric I4 is two's-complement in the checkpoint. XOR 8
        # maps it to the uint4b8 representation expected by the kernel.
        logical = torch.stack(
            [((source >> shift) & 0xF) ^ 0x8 for shift in source_shifts], dim=-1
        ).reshape(input_size, -1)
        logical = logical[:, : output_end - output_start].t().contiguous()
        groups = logical.reshape(output_end - output_start, input_size // 8, 8)

        destination = torch.zeros(
            (output_end - output_start, input_size // 8), dtype=torch.int32
        )
        for value_index, shift in enumerate(EXLLAMA_K_SHIFTS):
            destination.bitwise_or_(groups[:, :, value_index] << shift)
        packed_kernel_weight[output_start:output_end].copy_(destination)

    return packed_kernel_weight


def requantize_exllama_w4_to_dense_w3(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    output_chunk_size: int = 128,
) -> QronosW3A16DecodeWeights:
    """Build a compact decode-only INT3 shadow with per-group fitted scales."""
    _require_cpu_contiguous(packed_weight, "packed_weight")
    _require_cpu_contiguous(scales, "scales")
    if packed_weight.dtype is not torch.int32 or packed_weight.ndim != 2:
        raise ValueError("packed_weight must be contiguous CPU int32 [N,K/8]")
    if scales.dtype is not torch.bfloat16 or scales.ndim != 2:
        raise ValueError("3-bit decode scales require contiguous CPU BF16 [N,K/128]")
    output_size, packed_k = packed_weight.shape
    input_size = packed_k * 8
    num_groups = input_size // GROUP_SIZE
    if tuple(scales.shape) != (output_size, num_groups):
        raise ValueError(
            "3-bit source scales must match packed weights, got "
            f"{tuple(scales.shape)} for {(output_size, num_groups)}"
        )
    if input_size % 32:
        raise ValueError("3-bit dense packing requires K divisible by 32")
    if output_chunk_size <= 0:
        raise ValueError("output_chunk_size must be positive")

    packed_decode = torch.empty(
        (output_size, input_size * 3 // 32),
        dtype=torch.int32,
        device="cpu",
    )
    decode_scales = torch.empty_like(scales)
    candidate_alphas = (1.25, 1.5, 1.75, 2.0, 2.25, 2.5)

    for output_start in range(0, output_size, output_chunk_size):
        output_end = min(output_start + output_chunk_size, output_size)
        source = packed_weight[output_start:output_end]
        logical = torch.stack(
            [((source >> shift) & 0xF) - 8 for shift in EXLLAMA_K_SHIFTS],
            dim=-1,
        ).reshape(output_end - output_start, num_groups, GROUP_SIZE)
        logical_float = logical.to(torch.float32)

        best_error = torch.full(
            (output_end - output_start, num_groups),
            float("inf"),
            dtype=torch.float32,
        )
        best_alpha = torch.ones_like(best_error)
        best_q3 = torch.zeros_like(logical, dtype=torch.int8)
        for seed_alpha in candidate_alphas:
            q3 = torch.round(logical_float / seed_alpha).clamp(-4, 3)
            denominator = (q3 * q3).sum(dim=-1).clamp_min_(1.0)
            alpha = (logical_float * q3).sum(dim=-1) / denominator
            alpha.clamp_(min=0.5, max=4.0)
            error = (
                logical_float - q3 * alpha.unsqueeze(-1)
            ).square().sum(dim=-1)
            selected = error < best_error
            best_error = torch.where(selected, error, best_error)
            best_alpha = torch.where(selected, alpha, best_alpha)
            best_q3 = torch.where(
                selected.unsqueeze(-1),
                q3.to(torch.int8),
                best_q3,
            )

        decode_scales[output_start:output_end].copy_(
            (
                scales[output_start:output_end].float() * best_alpha
            ).to(torch.bfloat16)
        )
        biased = (best_q3.to(torch.int16) + 4).to(torch.uint8)
        groups_of_eight = biased.reshape(
            output_end - output_start, input_size // 8, 8
        )
        packed_24 = torch.zeros(
            (output_end - output_start, input_size // 8),
            dtype=torch.int32,
        )
        for value_index in range(8):
            packed_24.bitwise_or_(
                groups_of_eight[:, :, value_index].to(torch.int32)
                << (3 * value_index)
            )
        packed_bytes = torch.stack(
            (
                packed_24.to(torch.uint8),
                (packed_24 >> 8).to(torch.uint8),
                (packed_24 >> 16).to(torch.uint8),
            ),
            dim=-1,
        ).reshape(output_end - output_start, input_size * 3 // 8)
        packed_decode[output_start:output_end].copy_(
            packed_bytes.contiguous().view(torch.int32)
        )

    return QronosW3A16DecodeWeights(
        packed_weight=packed_decode,
        scales=decode_scales,
    )


def _transform_quark_w4a16(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    packed_zero_points: torch.Tensor,
    *,
    padded_output_size: int | None = None,
    group_size: int = GROUP_SIZE,
    output_chunk_size: int = 256,
    expected_scale_dtype: torch.dtype,
    format_name: str,
    kernel_scale_dtype: torch.dtype,
) -> QronosW4A16Weights:
    """Validate and transform one checkpoint-native Quark linear tensor set."""
    input_size, output_size = _validate_checkpoint_tensors(
        packed_weight,
        scales,
        packed_zero_points,
        group_size,
        expected_scale_dtype=expected_scale_dtype,
        format_name=format_name,
    )
    if padded_output_size is None:
        padded_output_size = (output_size + 7) // 8 * 8
    if kernel_scale_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "kernel_scale_dtype must be torch.float32 or torch.bfloat16, "
            f"got {kernel_scale_dtype}"
        )
    packed_kernel_weight = repack_qronos_reorder_to_exllama(
        packed_weight,
        output_size,
        padded_output_size,
        output_chunk_size=output_chunk_size,
    )
    kernel_scales = torch.zeros(
        (padded_output_size, input_size // group_size), dtype=kernel_scale_dtype
    )
    # ``copy_`` performs the requested F32->BF16 rounding while transposing.
    # This output is the only scale tensor the caller may transfer to the GPU;
    # the checkpoint's F32 mmap tensor remains CPU-only.
    kernel_scales[:output_size].copy_(scales.t())
    return QronosW4A16Weights(
        packed_weight=packed_kernel_weight,
        scales=kernel_scales,
        input_size=input_size,
        output_size=output_size,
        padded_output_size=padded_output_size,
        group_size=group_size,
    )


def transform_qronos_w4a16(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    packed_zero_points: torch.Tensor,
    *,
    padded_output_size: int | None = None,
    group_size: int = GROUP_SIZE,
    output_chunk_size: int = 256,
    kernel_scale_dtype: torch.dtype = torch.float32,
) -> QronosW4A16Weights:
    """Transform one Qronos linear; checkpoint scales must be F32."""
    return _transform_quark_w4a16(
        packed_weight,
        scales,
        packed_zero_points,
        padded_output_size=padded_output_size,
        group_size=group_size,
        output_chunk_size=output_chunk_size,
        expected_scale_dtype=torch.float32,
        format_name="Qronos",
        kernel_scale_dtype=kernel_scale_dtype,
    )


def transform_awq_w4a16(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    packed_zero_points: torch.Tensor,
    *,
    padded_output_size: int | None = None,
    group_size: int = GROUP_SIZE,
    output_chunk_size: int = 256,
    kernel_scale_dtype: torch.dtype = torch.float32,
) -> QronosW4A16Weights:
    """Transform one AMD Quark AWQ linear; checkpoint scales must be BF16."""
    return _transform_quark_w4a16(
        packed_weight,
        scales,
        packed_zero_points,
        padded_output_size=padded_output_size,
        group_size=group_size,
        output_chunk_size=output_chunk_size,
        expected_scale_dtype=torch.bfloat16,
        format_name="Quark AWQ",
        kernel_scale_dtype=kernel_scale_dtype,
    )
