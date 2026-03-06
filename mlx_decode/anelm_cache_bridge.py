"""
Bridge ANE-LM binary cache file to mlx_lm's cache format for Qwen3.5.

ANE-LM exports its KV/SSM/conv state via the `ane-lm prefill` subcommand.
The binary file format is:

    [16 bytes] magic: b"ANELM_CACHE\n\x00\x00\x00\x00"
    [4 bytes]  uint32: JSON header length
    [N bytes]  JSON: model config (num_layers, num_kv_heads, head_dim, ...)
    For each layer:
      [1 byte]   type: 0=LinearAttention, 1=FullAttention
      If linear:
        [4 bytes]  int32: conv_pos (circular buffer position, oldest slot)
        [float32 * lin_qkv_dim * (conv_kernel-1)]  conv_state  [C, K-1]
        [float32 * lin_num_val_heads * lin_key_dim * lin_val_dim]  ssm_state [Hv, Dk, Dv]
      If full_attn:
        [float32 * prompt_len * num_kv_heads * head_dim]  k_cache  [N, Hkv, D]
        [float32 * prompt_len * num_kv_heads * head_dim]  v_cache  [N, Hkv, D]

Shape conversions required for MLX:
    KV:  ANE [N, Hkv, D]    -> MLX [1, Hkv, N, D]       (transpose + batch dim)
    SSM: ANE [Hv, Dk, Dv]   -> MLX [1, Hv, Dv, Dk]      (swap last two dims + batch dim)
    conv: ANE [C, K-1] circ -> MLX [1, K-1, C]           (unroll circular buf + transpose + batch dim)
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

MAGIC = b"ANELM_CACHE\n\x00\x00\x00\x00"


def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"Expected {n} bytes, got {len(data)}")
    return data


def _is_linear_layer(layer_idx: int, full_attention_interval: int) -> bool:
    """Mirror the logic in mlx_lm's Qwen3.5 DecoderLayer."""
    return (layer_idx + 1) % full_attention_interval != 0


def load_anelm_cache(path: str | Path, config: Dict[str, Any]) -> Tuple[List[Any], int]:
    """Read an ANE-LM binary cache file and return MLX-format cache objects.

    Args:
        path: Path to the .bin cache file produced by `ane-lm prefill`.
        config: Model config dict (top-level or with text_config sub-dict).

    Returns:
        (caches, prompt_len):
            caches     -- list of cache objects (KVCache / ArraysCache) per layer
            prompt_len -- number of tokens that were prefilled
    """
    text_cfg = config.get("text_config", config)
    full_attn_interval: int = text_cfg.get("full_attention_interval", 4)

    with open(path, "rb") as f:
        # --- Magic ---
        magic = _read_exact(f, 16)
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")

        # --- JSON header ---
        (json_len,) = struct.unpack("<I", _read_exact(f, 4))
        meta = json.loads(_read_exact(f, json_len).decode())

        num_layers: int  = meta["num_layers"]
        prompt_len: int  = meta["prompt_len"]
        num_kv_heads: int = meta["num_kv_heads"]
        head_dim: int    = meta["head_dim"]
        lin_num_val_heads: int = meta["lin_num_val_heads"]
        lin_key_dim: int  = meta["lin_key_dim"]
        lin_val_dim: int  = meta["lin_val_dim"]
        lin_qkv_dim: int  = meta["lin_qkv_dim"]
        conv_kernel: int  = meta["conv_kernel"]

        caches: List[Any] = []

        for L in range(num_layers):
            (layer_type,) = struct.unpack("B", _read_exact(f, 1))

            if layer_type == 0:  # LinearAttention (DeltaNet)
                (conv_pos,) = struct.unpack("<i", _read_exact(f, 4))

                # conv_state: ANE stores [C, K-1] as circular buffer
                # where slot conv_pos is the oldest, (conv_pos+1)%(K-1) is next, etc.
                K_minus_1 = conv_kernel - 1
                conv_flat = np.frombuffer(
                    _read_exact(f, lin_qkv_dim * K_minus_1 * 4), dtype=np.float32
                ).reshape(lin_qkv_dim, K_minus_1).copy()
                # Unroll circular buffer: reorder slots oldest → newest
                # slot conv_pos is oldest
                order = [(conv_pos + k) % K_minus_1 for k in range(K_minus_1)]
                conv_reordered = conv_flat[:, order]   # [C, K-1], oldest→newest
                # MLX wants [1, K-1, C]
                conv_mlx = conv_reordered.T[np.newaxis, :, :]  # [1, K-1, C]

                # ssm_state: ANE stores [Hv, Dk, Dv]
                # MLX wants [1, Hv, Dv, Dk]
                ssm_flat = np.frombuffer(
                    _read_exact(f, lin_num_val_heads * lin_key_dim * lin_val_dim * 4),
                    dtype=np.float32,
                ).reshape(lin_num_val_heads, lin_key_dim, lin_val_dim).copy()
                # Swap last two dims: [Hv, Dk, Dv] -> [Hv, Dv, Dk]
                ssm_mlx = np.swapaxes(ssm_flat, 1, 2)[np.newaxis, :, :, :]  # [1, Hv, Dv, Dk]

                cache = ArraysCache(size=2)
                cache[0] = mx.array(conv_mlx)
                cache[1] = mx.array(ssm_mlx)

            else:  # FullAttention (layer_type == 1)
                kv_step = num_kv_heads * head_dim
                k_flat = np.frombuffer(
                    _read_exact(f, prompt_len * kv_step * 4), dtype=np.float32
                ).reshape(prompt_len, num_kv_heads, head_dim).copy()
                v_flat = np.frombuffer(
                    _read_exact(f, prompt_len * kv_step * 4), dtype=np.float32
                ).reshape(prompt_len, num_kv_heads, head_dim).copy()

                # ANE [N, Hkv, D] -> MLX [1, Hkv, N, D]
                k_mlx = np.transpose(k_flat, (1, 0, 2))[np.newaxis, :, :, :]
                v_mlx = np.transpose(v_flat, (1, 0, 2))[np.newaxis, :, :, :]

                cache = KVCache()
                cache.keys = mx.array(k_mlx)
                cache.values = mx.array(v_mlx)
                cache.offset = prompt_len

            caches.append(cache)

    # Force evaluation of all lazy arrays
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

    return caches, prompt_len
