"""
Bridge CoreML prefill output states to mlx_lm's cache format for Qwen3.5.

Qwen3.5 uses hybrid attention: 3/4 layers are DeltaNet (linear attention),
1/4 are full attention. CoreML prefill outputs flattened named tensors that
must be converted to mlx_lm's cache objects for GPU decode.

Cache type per layer (determined by layer index):
  - Linear attention (DeltaNet): ArraysCache(size=2)
      cache[0] = conv_state:  [B, conv_kernel_size-1, conv_dim]
      cache[1] = delta_state: [B, Hv, Dv, Dk]
  - Full attention:            KVCache
      keys:   [B, num_kv_heads, seq_len, head_dim]
      values: [B, num_kv_heads, seq_len, head_dim]
      offset: prompt_len (critical for RoPE position tracking)

CoreML outputs numpy arrays with HF-convention names and shapes.
This module handles the format conversion, including the conv_state
transpose required between HF format and MLX format.
"""

from typing import Any, Dict, List

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache


def _is_linear_layer(layer_idx: int, full_attention_interval: int) -> bool:
    """Return True if the given layer uses linear (DeltaNet) attention.

    Mirrors the logic in mlx_lm's Qwen3.5 DecoderLayer:
        self.is_linear = (layer_idx + 1) % full_attention_interval != 0
    """
    return (layer_idx + 1) % full_attention_interval != 0


def _get_text_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the text_config sub-dict, falling back to config itself.

    The top-level config for Qwen3.5 VLM wraps language model parameters
    under a "text_config" key.  Pure language model configs may not have
    this wrapper.
    """
    return config.get("text_config", config)


def coreml_to_mlx_cache(
    coreml_outputs: Dict[str, Any],
    config: Dict[str, Any],
    prompt_len: int,
) -> List[Any]:
    """Convert CoreML prefill output tensors to a list of mlx_lm cache objects.

    Args:
        coreml_outputs: Dict of named numpy arrays produced by CoreML prefill.
            Expected keys for full-attention layer *i*:
                ``past_key_{i}``   -- shape [1, num_kv_heads, seq_len, head_dim]
                ``past_value_{i}`` -- shape [1, num_kv_heads, seq_len, head_dim]
            Expected keys for linear-attention (DeltaNet) layer *i*:
                ``conv_state_{i}``      -- shape [1, conv_dim, conv_kernel_size]  (HF format)
                ``recurrent_state_{i}`` -- shape [1, Hv, Dv, Dk]
        config: Model config dict (may contain a ``text_config`` sub-dict).
        prompt_len: Number of real (non-padding) prompt tokens that were
            processed during prefill.  Used to set ``KVCache.offset`` so
            that subsequent RoPE positions are correct.

    Returns:
        A list of cache objects (one per layer) matching the format produced
        by ``mlx_lm``'s ``Model.make_cache()`` for Qwen3.5:
            - ``ArraysCache(size=2)`` for DeltaNet layers
            - ``KVCache()`` for full-attention layers
    """
    text_cfg = _get_text_config(config)
    num_layers: int = text_cfg["num_hidden_layers"]
    full_attn_interval: int = text_cfg.get("full_attention_interval", 4)

    caches: List[Any] = []

    for i in range(num_layers):
        if _is_linear_layer(i, full_attn_interval):
            cache = _build_arrays_cache(coreml_outputs, i)
        else:
            cache = _build_kv_cache(coreml_outputs, i, prompt_len)
        caches.append(cache)

    # Force evaluation of all lazy mlx arrays so downstream decode is not
    # blocked by pending transfers from numpy.
    all_arrays: List[mx.array] = []
    for c in caches:
        if isinstance(c, KVCache):
            if c.keys is not None:
                all_arrays.extend([c.keys, c.values])
        elif isinstance(c, ArraysCache):
            for slot in c.cache:
                if slot is not None:
                    all_arrays.append(slot)
    if all_arrays:
        mx.eval(*all_arrays)

    return caches


def _build_kv_cache(
    outputs: Dict[str, Any],
    layer_idx: int,
    prompt_len: int,
) -> KVCache:
    """Construct a KVCache for a full-attention layer from CoreML outputs.

    CoreML provides keys/values in the standard HF shape:
        [B, num_kv_heads, seq_len, head_dim]
    which matches what mlx_lm's KVCache expects internally.

    Setting ``offset = prompt_len`` is critical: KVCache.offset controls the
    RoPE position counter.  After prefilling *prompt_len* tokens, the next
    decode step must use position *prompt_len* (0-indexed), which is exactly
    what KVCache computes as ``prev = self.offset`` in ``update_and_fetch``.
    """
    key_name = f"past_key_{layer_idx}"
    value_name = f"past_value_{layer_idx}"

    keys = mx.array(outputs[key_name])
    values = mx.array(outputs[value_name])

    cache = KVCache()
    # Directly assign the pre-allocated key/value buffers.
    cache.keys = keys
    cache.values = values
    # offset must equal prompt_len so that RoPE positions continue correctly.
    cache.offset = prompt_len

    return cache


def _build_arrays_cache(
    outputs: Dict[str, Any],
    layer_idx: int,
) -> ArraysCache:
    """Construct an ArraysCache for a DeltaNet layer from CoreML outputs.

    ArraysCache(size=2) stores two arrays:
        cache[0] = conv_state:      [B, conv_kernel_size - 1, conv_dim]
        cache[1] = recurrent_state: [B, Hv, Dv, Dk]

    CoreML outputs conv_state in HF format:
        [B, conv_dim, conv_kernel_size]
    Conversion requires:
        1. Transpose the last two dimensions -> [B, conv_kernel_size, conv_dim]
        2. Keep only the last (conv_kernel_size - 1) timesteps
           -> [B, conv_kernel_size - 1, conv_dim]

    The recurrent state is already in the correct shape [B, Hv, Dv, Dk] and
    requires no transformation.
    """
    conv_state_name = f"conv_state_{layer_idx}"
    recurrent_state_name = f"recurrent_state_{layer_idx}"

    # --- conv_state: HF -> MLX format ---
    # HF:  [B, conv_dim, conv_kernel_size]  e.g. [1, 6144, 4]
    # MLX: [B, conv_kernel_size - 1, conv_dim]  e.g. [1, 3, 6144]
    conv_state_hf = mx.array(outputs[conv_state_name])
    # Transpose last two dims: [B, conv_dim, K] -> [B, K, conv_dim]
    conv_state_transposed = mx.swapaxes(conv_state_hf, -2, -1)
    # Take only the last (K - 1) timesteps: [B, K, conv_dim] -> [B, K-1, conv_dim]
    conv_state_mlx = conv_state_transposed[:, 1:, :]

    # --- recurrent_state: no conversion needed ---
    # Shape: [B, Hv, Dv, Dk]  (same in both HF and MLX)
    recurrent_state = mx.array(outputs[recurrent_state_name])

    cache = ArraysCache(size=2)
    cache[0] = conv_state_mlx
    cache[1] = recurrent_state

    return cache


def make_empty_cache(config: Dict[str, Any]) -> List[Any]:
    """Create an empty cache list matching mlx_lm's make_cache() for Qwen3.5.

    This produces the same structure as ``Model.make_cache()`` in
    ``mlx_lm/models/qwen3_5.py``:
        [ArraysCache(size=2) if is_linear else KVCache() for each layer]

    Useful for starting decode from scratch (without a CoreML prefill step).

    Args:
        config: Model config dict (may contain a ``text_config`` sub-dict).

    Returns:
        A list of empty cache objects, one per decoder layer.
    """
    text_cfg = _get_text_config(config)
    num_layers: int = text_cfg["num_hidden_layers"]
    full_attn_interval: int = text_cfg.get("full_attention_interval", 4)

    caches: List[Any] = []
    for i in range(num_layers):
        if _is_linear_layer(i, full_attn_interval):
            caches.append(ArraysCache(size=2))
        else:
            caches.append(KVCache())
    return caches
