# Modifications

This package modifies and combines third-party components. The statements below
are the prominent notices required by the retained licences.

## comfy-kitchen v0.2.33 (Apache-2.0)

File/derivation: `minimax_h3_projections_gfx1201.so` contains a modified D128
INT8 attention kernel and fragment helpers derived from
`Comfy-Org/comfy-kitchen` revision `e9ea99cf2f0af1d0c49c04690d4153a91c2b8668`,
Copyright (c) 2025 Comfy Org.

Changes:

- prepacked K/V fragments for the 56-head, D128 H3 layout;
- query tiling for the packed Ref2VA sequence;
- native U8 packing and instruction scheduling for gfx1201;
- fused residual/SwiGLU/Turbo epilogues around the projection kernels.

Retained: the original attention quantization and online-softmax arithmetic.

## MiniMax H3 (MiniMax H3 Community License)

`paiton_video/` and the artifact implement an alternative execution path for
the upstream H3 graph. The model weights are not modified or redistributed by
this package; they are downloaded from the pinned upstream revisions in
`checkpoints.lock.json`. Modified runtime files carry this notice by inclusion
in the package. Ref2VA reference packing follows ComfyUI's `PackedLayout`
(document order `[text | refs | audio | video]`, shared RoPE clock).

## ComfyUI / ComfyUI frontend / AIMDO (GPL-3.0)

Not distributed with the published image. `Dockerfile.local` fetches the pinned
upstream sources and builds a local-only image for GUI parity. That image is not
a publication artifact and must not be redistributed without satisfying the GPL.
