"""Model loading.

Two optional accelerations are handled here: flash-attn_3 (backbone and audio encoder)
and liger (backbone RMSNorm / SwiGLU). Both are used when installed and fall back
with a warning otherwise. A fallback changes bf16 reduction order and therefore the
numerical output.
"""

import logging

import torch

from .configuration_fireredaudio import FireRedAudioConfig
from .modeling_fireredaudio import FireRedAudioForCausalLM

logger = logging.getLogger(__name__)


def load_fireredaudio(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    quantization: str | None = None,
) -> FireRedAudioForCausalLM:
    """Load the model for inference.

    Args:
        model_name_or_path: Directory holding config.json and safetensors shards.
        dtype: Weight dtype; bfloat16 matches the released weights.
        device: Moved there when given.
        quantization: Optional ``int4`` (bitsandbytes NF4). Quantization applies
            to supported linear layers.

    Returns:
        A FireRedAudioForCausalLM in eval mode.
    """
    config = FireRedAudioConfig.from_pretrained(model_name_or_path)
    attn = _resolve_attn()

    # dit, patch_encoder and vae/downsample hardcode their attention and ignore this.
    config.backbone_config._attn_implementation = attn
    config.audio_encoder_config._attn_implementation = attn
    config.red_vae_config._attn_implementation = attn

    _apply_liger()

    quantization_config = _make_quantization_config(quantization, dtype)
    load_kwargs = {
        "config": config,
        "torch_dtype": dtype,
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
        if device is not None:
            if torch.device(device).type != "cuda":
                raise ValueError("quantized loading requires a CUDA device")
            load_kwargs["device_map"] = {"": str(device)}

    model = FireRedAudioForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    model.eval()
    if device is not None and quantization_config is None:
        model.to(device)
    return model


def _make_quantization_config(quantization: str | None, dtype: torch.dtype):
    if quantization is None:
        return None
    if quantization == "int4":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "int4 quantization requires bitsandbytes; install the quantization "
                "extras before using quantization='int4'"
            ) from exc
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError("quantization must be one of: None, 'int4'")


def _resolve_attn() -> str:
    """flash_attention_3 when flash-attn_3 is installed, otherwise sdpa."""
    try:
        import flash_attn_3  # noqa: F401
    except ImportError:
        logger.warning(
            "flash-attn_3 not installed; falling back to sdpa, which changes the "
            "numerical output."
        )
        return "sdpa"
    return "flash_attention_3"


def _apply_liger() -> bool:
    """Enable liger RMSNorm / SwiGLU if available; returns whether it was applied.

    liger differs from the stock implementation in bf16 reduction order, and that
    difference accumulates over 32 layers. This monkeypatches
    transformers.models.qwen3_5 globally and cannot be undone within a process.
    """
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
    except ImportError:
        logger.warning(
            "liger_kernel not installed; RMSNorm/SwiGLU fall back to the stock transformers "
            "implementation, which changes the numerical output."
        )
        return False

    apply_liger_kernel_to_qwen3_5(
        rope=False,
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )
    return True
