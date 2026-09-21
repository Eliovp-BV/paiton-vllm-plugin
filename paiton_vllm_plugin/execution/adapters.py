"""Installed external adapters. Merely reading this table imports no runtime."""

MODELS = {
    "minicpm5-awq-v1": ("PaitonMiniCPM5AWQForCausalLM", "paiton_minicpm5_awq"),
    "qwen3-coder-v1": ("PaitonQwen3CoderForCausalLM", "paiton_qwen3_coder"),
    "gptoss-mxfp4-v1": ("PaitonGptOssForCausalLM", "paiton_gptoss"),
    "qwen38-qronos-v1": ("PaitonQwen38ForCausalLM", "paiton_qwen38"),
    "ornith15-v1": ("PaitonOrnith15ForCausalLM", "paiton_ornith15"),
    "qwen38-neo-v1": (
        "PaitonQwen38GGUFForConditionalGeneration",
        "paiton_qwen38_gguf_multimodal",
    ),
}
PLATFORM = {"qwen38-qronos-v1", "ornith15-v1", "qwen38-neo-v1"}
