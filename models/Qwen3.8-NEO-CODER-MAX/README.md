# Qwen3.8 NEO CODER MAX qualification candidate

**Not a qualified release. No Paiton serving image is published for this entry.**
The files in this directory pin the requested fine-tune for implementation and
qualification; they do not add it to the supported community release table.

The requested [author GGUF repository](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF/tree/89230607b3708bc1efe174e8fd34d4b164476c88)
and [author-linked source](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU/tree/895782677541896947ea136714b45dbed994daca)
are separate pinned inputs. Their revisions and selected file hashes are in
[checkpoint.lock.json](checkpoint.lock.json).

The selected MTP Q4_K_M file is 18,498,573,856 bytes. It contains 433 Q4_K,
64 Q6_K, eight Q8_0, 360 FP32 and one BF16 tensor. The BF16 tensor is the output
projection. The 24,033,703,456-byte MTP Q6_K alternative has had its metadata
inspected; it has not been qualified. Neither file is Quark/Qronos/AWQ INT4.

The architecture uses Qwen3.5 compatibility identifiers: 64 target layers,
48 GDN layers, 16 full-attention layers, and one additional MTP layer. This
matches the dimensions of Paiton's existing Qwen3.8 graph, but not its weight
loading or precision contract. The existing Qronos release and its optional
W4 output projection must not be substituted for this model.

The implementation route under evaluation preserves the author's GGUF tensor
values in native execution. Quantizing the author-source checkpoint for a
separate hardware profile remains an alternative; such a result would be a
separately named derivative, not an equivalent conversion of the NEO GGUF.

## Remaining qualification work

- Complete native graph and vLLM loader integration for the mixed weight
  formats, including the stored FP32 normalization and state coefficients.
- Validate full-model layer outputs, logits, generation, state isolation,
  cancellation, prefix reuse and the supported request profile.
- Complete native GDN prefill optimization and measure full request performance.
- Integrate MTP draft execution with verified acceptance/rejection and KV/GDN
  commit/rollback. Upstream MTP weights alone do not establish Paiton support.
- Qualify vision with a pinned compatible projector, or explicitly release a
  text-only profile. Vision is currently unqualified.
- Freeze and verify the model artifact, runtime ABI, licenses, hashes, SBOM,
  provenance and public artifact allowlist; inspect all final image layers.
- Publish only after these checks, then pull the immutable digest and validate
  it without a compiler checkout.

There is no supported generation launch command for this candidate yet. To
inspect an already downloaded checkpoint without loading its weights:

```bash
python3 paiton_vllm_plugin/gguf_inventory.py /path/to/model.gguf --output /path/to/inventory.json
```

Keep inventory output, downloads and experimental builds outside the repository.
The upstream repositories declare Apache-2.0; publication still requires a
review of notices for the final linked runtime and model artifacts. No model
weights or compiler artifacts are distributed in this candidate directory.
Existing community tags, launchers and model defaults are unchanged.
