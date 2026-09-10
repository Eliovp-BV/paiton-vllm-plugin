# License and provenance

- GPT-OSS-20B weights and chat template: OpenAI, Apache-2.0, revision
  `6cee5e81ee83917806bbde320786a8fb61efebee`. The upstream `MODEL_LICENSE`
  and `USAGE_POLICY` are included. Weights are downloaded into persistent
  user storage; they are not embedded in the image.
- Runtime integration and benchmark scripts use this repository's Apache-2.0
  license. The overlay contains the compiled Paiton artifact and its manifest.
  Private compiler source is excluded.
- The compiled runtime includes AITemplate-derived components, Apache-2.0,
  and links to the ROCm libraries from the pinned base image. Existing repository
  third-party notices and license texts remain applicable.
- The exact vLLM/PyTorch/Triton/Transformers/ROCm runtime comes from
  `rocm/vllm-dev@sha256:4fa5bc9c24d25ef7e0fe38b23fc5f2b27fc986177bd2f9e496997bcf5af03871`.
  Their individual license files and notices remain in the runtime payload.
- The benchmark-only chat template freezes the date at 2026-09-09. Normal
  serving uses the unchanged upstream template and the current date.

- Harmony vocabulary: OpenAI tiktoken `o200k_base.tiktoken`, MIT; see
  `TIKTOKEN_LICENSE`. The Dockerfile pins the upstream content SHA-256 and
  embeds the vocabulary for offline chat.
