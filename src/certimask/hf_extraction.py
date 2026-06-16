"""Extract post-RoPE Q/K and V from Hugging Face Qwen2/Qwen2.5 models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import transformers  # type: ignore[import-untyped]

    TRANSFORMERS_VERSION: str | None = transformers.__version__
except ImportError:
    TRANSFORMERS_VERSION = None

SUPPORTED_ARCHITECTURES = {"Qwen2", "Qwen2ForCausalLM", "Qwen2ForSequenceClassification"}


@dataclass
class ExtractedQK:
    """Post-RoPE Query and Key tensors extracted from a model.

    Attributes:
        query: Post-RoPE query, shape [B, num_query_heads, L, head_dim].
        key: Post-RoPE key, shape [B, num_kv_heads, L, head_dim].
        layer_index: Which decoder layer was extracted from.
        num_query_heads: Number of query heads.
        num_key_value_heads: Number of key/value heads.
        head_dim: Head dimension.
        sequence_length: Sequence length.
        model_name: Model name or path.
        transformers_version: Version of transformers used.
    """

    query: torch.Tensor
    key: torch.Tensor
    layer_index: int
    num_query_heads: int
    num_key_value_heads: int
    head_dim: int
    sequence_length: int
    model_name: str
    transformers_version: str


@dataclass
class ExtractedQKV:
    """Post-RoPE Q/K and V tensors extracted from a model.

    Attributes:
        query: Post-RoPE query, shape [B, num_query_heads, L, head_dim].
        key: Post-RoPE key, shape [B, num_kv_heads, L, head_dim].
        value: Value states, shape [B, num_kv_heads, L, head_dim].
        layer_index: Which decoder layer was extracted from.
        num_query_heads: Number of query heads.
        num_key_value_heads: Number of key/value heads.
        head_dim: Head dimension.
        sequence_length: Sequence length.
        model_name: Model name or path.
        transformers_version: Version of transformers used.
    """

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    layer_index: int
    num_query_heads: int
    num_key_value_heads: int
    head_dim: int
    sequence_length: int
    model_name: str
    transformers_version: str


def extract_qk_from_qwen2(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    layer_index: int = 0,
) -> ExtractedQK:
    """Extract post-RoPE Q/K from a Qwen2/Qwen2.5 model at a specific layer.

    Uses a forward hook on the target decoder layer to capture hidden_states,
    then manually runs q_proj, k_proj, and applies RoPE using the model's
    own rotary embedding.

    Args:
        model: A Qwen2ForCausalLM or Qwen2Model instance.
        input_ids: Input token IDs, shape [B, L].
        attention_mask: Optional attention mask, shape [B, L].
        layer_index: Decoder layer index to extract from.

    Returns:
        ExtractedQK with post-RoPE Q/K tensors and metadata.

    Raises:
        ImportError: If transformers is not installed.
        ValueError: If model architecture is not supported or parameters are invalid.
    """
    if TRANSFORMERS_VERSION is None:
        raise ImportError("transformers is required. Install with: pip install certimask[hf]")

    _validate_model(model)
    _validate_inputs(input_ids, attention_mask, model, layer_index)

    model.eval()

    # Get the model internals — Qwen2ForCausalLM wraps Qwen2Model
    base_model = model.model if hasattr(model, "model") else model

    decoder_layer = base_model.layers[layer_index]
    attn_module = decoder_layer.self_attn

    # IMPORTANT: Qwen2DecoderLayer applies input_layernorm BEFORE self_attn.
    # We must hook into self_attn (not the decoder layer) to capture
    # post-layernorm hidden_states that match what q_proj/k_proj receive.
    captured_hidden_states: list[torch.Tensor] = []
    captured_cos_sin: list[tuple[torch.Tensor, torch.Tensor]] = []

    def attn_pre_hook(
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        # In transformers 4.49, Qwen2Attention.forward receives
        # hidden_states as first kwarg
        hs = kwargs.get("hidden_states")
        if isinstance(hs, torch.Tensor):
            captured_hidden_states.append(hs.detach())
        pos_emb = kwargs.get("position_embeddings")
        if isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            captured_cos_sin.append((pos_emb[0].detach(), pos_emb[1].detach()))

    hook = attn_module.register_forward_pre_hook(attn_pre_hook, with_kwargs=True)

    # Run the model forward to trigger the hook
    with torch.no_grad():
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = None

        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
        )

    hook.remove()

    if not captured_hidden_states:
        raise RuntimeError("Hook did not capture hidden states. Model forward may have failed.")

    hidden_states = captured_hidden_states[0]

    # Get cos/sin from the captured position embeddings or compute them
    if captured_cos_sin:
        cos, sin = captured_cos_sin[0]
    else:
        seq_length = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
        cos, sin = base_model.rotary_emb(hidden_states, position_ids)

    # Manually compute Q and K: q_proj -> reshape -> apply_rope
    with torch.no_grad():
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_module.head_dim)

        query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        from transformers.models.qwen2.modeling_qwen2 import (  # type: ignore[import-untyped]
            apply_rotary_pos_emb,
        )

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Get model metadata
    config = model.config if hasattr(model, "config") else base_model.config
    model_name = getattr(config, "_name_or_path", "unknown")

    return ExtractedQK(
        query=query_states,
        key=key_states,
        layer_index=layer_index,
        num_query_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        sequence_length=hidden_states.shape[1],
        model_name=model_name,
        transformers_version=TRANSFORMERS_VERSION,
    )


def extract_qkv_from_qwen2(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    layer_index: int = 0,
) -> ExtractedQKV:
    """Extract post-RoPE Q/K and V from a Qwen2/Qwen2.5 model.

    Uses the same hook strategy as extract_qk_from_qwen2, additionally
    computing v_proj on the captured hidden states.

    Args:
        model: A Qwen2ForCausalLM or Qwen2Model instance.
        input_ids: Input token IDs, shape [B, L].
        attention_mask: Optional attention mask, shape [B, L].
        layer_index: Decoder layer index to extract from.

    Returns:
        ExtractedQKV with post-RoPE Q/K and V tensors.
    """
    if TRANSFORMERS_VERSION is None:
        raise ImportError("transformers is required. Install with: pip install certimask[hf]")

    _validate_model(model)
    _validate_inputs(input_ids, attention_mask, model, layer_index)

    model.eval()

    base_model = model.model if hasattr(model, "model") else model
    decoder_layer = base_model.layers[layer_index]
    attn_module = decoder_layer.self_attn

    captured_hidden_states: list[torch.Tensor] = []
    captured_cos_sin: list[tuple[torch.Tensor, torch.Tensor]] = []

    def attn_pre_hook(
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        hs = kwargs.get("hidden_states")
        if isinstance(hs, torch.Tensor):
            captured_hidden_states.append(hs.detach())
        pos_emb = kwargs.get("position_embeddings")
        if isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            captured_cos_sin.append((pos_emb[0].detach(), pos_emb[1].detach()))

    hook = attn_module.register_forward_pre_hook(attn_pre_hook, with_kwargs=True)

    with torch.no_grad():
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = None

        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
        )

    hook.remove()

    if not captured_hidden_states:
        raise RuntimeError("Hook did not capture hidden states.")

    hidden_states = captured_hidden_states[0]

    if captured_cos_sin:
        cos, sin = captured_cos_sin[0]
    else:
        seq_length = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
        cos, sin = base_model.rotary_emb(hidden_states, position_ids)

    with torch.no_grad():
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_module.head_dim)

        query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        from transformers.models.qwen2.modeling_qwen2 import (
            apply_rotary_pos_emb,
        )

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    config = model.config if hasattr(model, "config") else base_model.config
    model_name = getattr(config, "_name_or_path", "unknown")

    return ExtractedQKV(
        query=query_states,
        key=key_states,
        value=value_states,
        layer_index=layer_index,
        num_query_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        sequence_length=hidden_states.shape[1],
        model_name=model_name,
        transformers_version=TRANSFORMERS_VERSION,
    )


def _validate_model(model: torch.nn.Module) -> None:
    """Validate that the model is a supported Qwen2 architecture.

    Raises:
        ValueError: If the model architecture is not supported.
    """
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model has no config attribute")

    model_type = getattr(config, "model_type", "")
    arch = getattr(config, "architectures", [])

    # Check model_type or architectures
    is_qwen2 = model_type in ("qwen2",) or any(
        a in SUPPORTED_ARCHITECTURES for a in arch
    )

    if not is_qwen2:
        raise ValueError(
            f"Unsupported model architecture: model_type={model_type}, "
            f"architectures={arch}. Expected Qwen2/Qwen2.5."
        )

    # Check required attributes
    if not hasattr(model, "model") and not hasattr(model, "layers"):
        raise ValueError(
            "Model must be Qwen2ForCausalLM or Qwen2Model with .model.layers"
        )

    base = model.model if hasattr(model, "model") else model
    if not hasattr(base, "layers"):
        raise ValueError("Base model has no .layers attribute")
    if not hasattr(base, "rotary_emb"):
        raise ValueError("Base model has no .rotary_emb attribute")


def _validate_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    model: torch.nn.Module,
    layer_index: int,
) -> None:
    """Validate input tensors and layer index.

    Raises:
        ValueError: If inputs are invalid.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2-D [B, L], got {input_ids.dim()}-D")

    if input_ids.numel() == 0:
        raise ValueError("input_ids is empty")

    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} != "
                f"input_ids shape {input_ids.shape}"
            )
        if attention_mask.dtype != torch.long and attention_mask.dtype != torch.int:
            raise ValueError(
                f"attention_mask must be integer type, got {attention_mask.dtype}"
            )

    # Get number of layers
    base = model.model if hasattr(model, "model") else model

    num_layers = len(base.layers)
    if layer_index < 0 or layer_index >= num_layers:
        raise ValueError(
            f"layer_index {layer_index} out of range [0, {num_layers})"
        )
