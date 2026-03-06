"""ANE-compatible Qwen3.5 prefill model for CoreML conversion.

Rewrites Qwen3.5 forward pass to be coremltools-compatible:
  - Conv2d(kernel_size=1) for all Linear layers (ANE requirement)
  - RMSNorm via doubled-tensor + layer_norm trick (Anemll pattern)
  - DeltaNet recurrence is functional (no in-place ops)
  - Full attention with output gate, QK-norm, and M-RoPE

Usage::

    from convert.ane_qwen35 import ANEQwen35Prefill
    model = ANEQwen35Prefill(config)
    model.load_hf_weights(hf_model.state_dict())
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ANE-compatible RMSNorm (Anemll pattern)
# ---------------------------------------------------------------------------

class ANERMSNorm(nn.Module):
    """RMSNorm using cat([x, -x]) → layer_norm → slice trick.

    Qwen3.5 uses ``output * (1 + weight)`` with weight initialized to 0,
    NOT ``output * weight`` with weight initialized to 1 like LLaMA.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))  # Qwen3.5: zeros
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        doubled = torch.cat([x_f, -x_f], dim=-1)
        normed = F.layer_norm(doubled, (2 * self.dim,), eps=self.eps)
        normed = normed[..., :self.dim]
        # Qwen3.5: (1 + weight), not just weight
        result = normed * (1.0 + self.weight.float())
        return result.type_as(x)


class ANERMSNormGated(nn.Module):
    """Gated RMSNorm for DeltaNet — norm first, then weight, then SiLU gate.

    Unlike Qwen3_5RMSNorm (which uses zeros + ``1+weight``),
    Qwen3_5RMSNormGated uses ones + ``weight *`` and applies the gate AFTER norm.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))  # RMSNormGated: ones
        self.eps = eps
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """x, z: [B*S*num_heads, head_dim]. Order: norm → weight → gate."""
        # 1. RMSNorm (doubled-tensor trick)
        x_f = x.float()
        doubled = torch.cat([x_f, -x_f], dim=-1)
        normed = F.layer_norm(doubled, (2 * self.head_dim,), eps=self.eps)
        normed = normed[..., :self.head_dim]
        # 2. Weight multiplication (just weight, no 1+weight)
        result = self.weight * normed.type_as(x)
        # 3. Gate (SiLU)
        result = result * F.silu(z.float())
        return result.type_as(x)


# ---------------------------------------------------------------------------
# SwiGLU MLP (Conv2d)
# ---------------------------------------------------------------------------

class ANEMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Conv2d(hidden_size, intermediate_size, 1, bias=False)
        self.up_proj = nn.Conv2d(hidden_size, intermediate_size, 1, bias=False)
        self.down_proj = nn.Conv2d(intermediate_size, hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, S, H] -> [B, H, 1, S]
        h = x.permute(0, 2, 1).unsqueeze(2)
        out = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return out.squeeze(2).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# DeltaNet (linear attention) — functional recurrence
# ---------------------------------------------------------------------------

class ANEDeltaNet(nn.Module):
    """ANE-compatible DeltaNet layer.

    HF weight names: in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
                     conv1d, A_log, dt_bias, norm, out_proj
    """

    def __init__(
        self,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.conv_kernel_dim = conv_kernel_dim

        self.qk_dim = num_key_heads * key_head_dim
        self.v_dim = num_value_heads * value_head_dim
        self.conv_inner_dim = self.qk_dim * 2 + self.v_dim  # QKV combined
        self.heads_per_group = num_value_heads // num_key_heads

        # Projections (Conv2d for ANE)
        # HF: in_proj_qkv [conv_inner_dim, hidden]
        self.in_proj_qkv = nn.Conv2d(hidden_size, self.conv_inner_dim, 1, bias=False)
        # HF: in_proj_z [v_dim, hidden] — z gate bypass
        self.in_proj_z = nn.Conv2d(hidden_size, self.v_dim, 1, bias=False)
        # HF: in_proj_a [num_v_heads, hidden] — decay
        self.in_proj_a = nn.Conv2d(hidden_size, num_value_heads, 1, bias=False)
        # HF: in_proj_b [num_v_heads, hidden] — beta gate
        self.in_proj_b = nn.Conv2d(hidden_size, num_value_heads, 1, bias=False)

        # Depthwise conv1d (HF uses bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_inner_dim, self.conv_inner_dim,
            kernel_size=conv_kernel_dim,
            padding=conv_kernel_dim - 1,
            groups=self.conv_inner_dim,
            bias=False,
        )

        # Learnable parameters
        self.A_log = nn.Parameter(torch.zeros(num_value_heads))
        self.dt_bias = nn.Parameter(torch.ones(num_value_heads))

        # Per-head gated RMSNorm (weight is [value_head_dim])
        self.norm = ANERMSNormGated(value_head_dim, eps=eps)

        # Output projection
        self.out_proj = nn.Conv2d(self.v_dim, hidden_size, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = hidden_states.shape

        # --- Projections via Conv2d ---
        h = hidden_states.permute(0, 2, 1).unsqueeze(2)  # [B, H, 1, S]

        qkv_raw = self.in_proj_qkv(h).squeeze(2)   # [B, conv_inner_dim, S]
        z_raw = self.in_proj_z(h).squeeze(2)         # [B, v_dim, S]
        a_raw = self.in_proj_a(h).squeeze(2)         # [B, num_v_heads, S]
        b_raw = self.in_proj_b(h).squeeze(2)         # [B, num_v_heads, S]

        # --- Depthwise conv1d + SiLU ---
        qkv_conv = self.conv1d(qkv_raw)[:, :, :S]
        qkv_conv = F.silu(qkv_conv)

        # --- Split Q, K, V ---
        q_raw = qkv_conv[:, :self.qk_dim, :]                    # [B, qk_dim, S]
        k_raw = qkv_conv[:, self.qk_dim:self.qk_dim * 2, :]    # [B, qk_dim, S]
        v_raw = qkv_conv[:, self.qk_dim * 2:, :]                # [B, v_dim, S]

        # Reshape to [B, S, heads, head_dim]
        q = q_raw.permute(0, 2, 1).reshape(B, S, self.num_key_heads, self.key_head_dim)
        k = k_raw.permute(0, 2, 1).reshape(B, S, self.num_key_heads, self.key_head_dim)
        v = v_raw.permute(0, 2, 1).reshape(B, S, self.num_value_heads, self.value_head_dim)

        # GQA expansion: key_heads → value_heads
        if self.heads_per_group > 1:
            q = q.unsqueeze(3).expand(
                B, S, self.num_key_heads, self.heads_per_group, self.key_head_dim
            ).reshape(B, S, self.num_value_heads, self.key_head_dim)
            k = k.unsqueeze(3).expand(
                B, S, self.num_key_heads, self.heads_per_group, self.key_head_dim
            ).reshape(B, S, self.num_value_heads, self.key_head_dim)

        # L2 normalize Q and K (match HF l2norm: rsqrt(sum(x*x) + eps))
        q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)

        # Scale Q by 1/sqrt(key_head_dim) — HF applies this before recurrence
        scale = 1.0 / (self.key_head_dim ** 0.5)
        q = q * scale

        # --- Decay and gate ---
        a = a_raw.permute(0, 2, 1)  # [B, S, num_v_heads]
        b = b_raw.permute(0, 2, 1)  # [B, S, num_v_heads]

        g = -torch.exp(self.A_log) * F.softplus(a + self.dt_bias)  # decay
        beta = torch.sigmoid(b)

        # --- Gated delta recurrence (functional) ---
        state = torch.zeros(
            B, self.num_value_heads, self.value_head_dim, self.key_head_dim,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )

        outputs = []
        for t in range(S):
            q_t = q[:, t]      # [B, Hv, Dk]
            k_t = k[:, t]      # [B, Hv, Dk]
            v_t = v[:, t]      # [B, Hv, Dv]
            g_t = g[:, t]      # [B, Hv]
            beta_t = beta[:, t]  # [B, Hv]

            decay = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)
            state = state * decay

            kv_mem = torch.einsum('bhvk,bhk->bhv', state, k_t)
            delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)
            state = state + torch.einsum('bhk,bhv->bhvk', k_t, delta)

            o_t = torch.einsum('bhvk,bhk->bhv', state, q_t)
            outputs.append(o_t)

        # [B, S, Hv, Dv]
        y = torch.stack(outputs, dim=1)

        # --- Gated RMSNorm (per-head) ---
        z = z_raw.permute(0, 2, 1)  # [B, S, v_dim]
        # Reshape to per-head: [B*S*Hv, Dv]
        y_flat = y.reshape(-1, self.value_head_dim)
        z_flat = z.reshape(B, S, self.num_value_heads, self.value_head_dim).reshape(-1, self.value_head_dim)
        y_normed = self.norm(y_flat, z_flat)
        y_out = y_normed.reshape(B, S, self.v_dim)

        # --- Output projection ---
        y_conv = y_out.permute(0, 2, 1).unsqueeze(2)
        out = self.out_proj(y_conv).squeeze(2).permute(0, 2, 1)

        # Cache outputs
        conv_state = qkv_raw[:, :, -self.conv_kernel_dim:]  # [B, conv_inner_dim, kernel]
        return out, conv_state, state


# ---------------------------------------------------------------------------
# Full attention (with output gate, QK-norm, M-RoPE)
# ---------------------------------------------------------------------------

class ANEFullAttention(nn.Module):
    """ANE-compatible full softmax attention.

    Qwen3.5 uses:
    - q_proj outputs 2x (query + gate): [hidden, num_heads * head_dim * 2]
    - q_norm, k_norm (per-head RMSNorm on head_dim)
    - M-RoPE (partial_rotary_factor=0.25, interleaved)
    - Output gate: attn_output *= sigmoid(gate)
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_seq_len: int = 512,
        rope_theta: float = 10000000.0,
        partial_rotary_factor: float = 0.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_groups = num_attention_heads // num_key_value_heads
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.scale = head_dim ** -0.5

        # Q projects to 2x (query + gate)
        self.q_proj = nn.Conv2d(hidden_size, num_attention_heads * head_dim * 2, 1, bias=False)
        self.k_proj = nn.Conv2d(hidden_size, num_key_value_heads * head_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(hidden_size, num_key_value_heads * head_dim, 1, bias=False)
        self.o_proj = nn.Conv2d(num_attention_heads * head_dim, hidden_size, 1, bias=False)

        # QK normalization (per-head, dimension = head_dim)
        self.q_norm = ANERMSNorm(head_dim, eps=eps)
        self.k_norm = ANERMSNorm(head_dim, eps=eps)

        # Pre-compute RoPE cos/sin tables as buffers (avoids torch.outer in forward)
        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [S, D/2]
        self.register_buffer("rope_cos", torch.cos(freqs))
        self.register_buffer("rope_sin", torch.sin(freqs))

    def _apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply RoPE to rotary portion. x: [B, nh, S, hd]"""
        rot = x[..., :self.rotary_dim]
        pass_through = x[..., self.rotary_dim:]
        x1 = rot[..., 0::2]
        x2 = rot[..., 1::2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        rot_out = torch.stack([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1).flatten(-2)
        return torch.cat([rot_out, pass_through], dim=-1)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = hidden_states.shape

        # Conv2d projections
        h = hidden_states.permute(0, 2, 1).unsqueeze(2)
        q_raw = self.q_proj(h).squeeze(2).permute(0, 2, 1)  # [B, S, num_heads*hd*2]
        k_raw = self.k_proj(h).squeeze(2).permute(0, 2, 1)  # [B, S, num_kv*hd]
        v_raw = self.v_proj(h).squeeze(2).permute(0, 2, 1)  # [B, S, num_kv*hd]

        # Split Q into query + gate
        q_and_gate = q_raw.view(B, S, self.num_heads, self.head_dim * 2)
        q = q_and_gate[..., :self.head_dim]   # [B, S, nh, hd]
        gate = q_and_gate[..., self.head_dim:]  # [B, S, nh, hd]
        gate = gate.reshape(B, S, -1)  # [B, S, nh*hd]

        # QK norm (per-head)
        q = self.q_norm(q)  # norm on last dim (head_dim)
        k = k_raw.view(B, S, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)

        # Transpose for attention: [B, heads, S, hd]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v_raw.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (use pre-computed tables)
        cos = self.rope_cos[:S].to(hidden_states.dtype)
        sin = self.rope_sin[:S].to(hidden_states.dtype)
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)

        # GQA expand K, V
        if self.num_groups > 1:
            k = k.unsqueeze(2).expand(
                B, self.num_kv_heads, self.num_groups, S, self.head_dim
            ).reshape(B, self.num_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(
                B, self.num_kv_heads, self.num_groups, S, self.head_dim
            ).reshape(B, self.num_heads, S, self.head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask (static)
        causal = torch.triu(
            torch.full((S, S), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1,
        )
        attn_weights = attn_weights + causal.unsqueeze(0).unsqueeze(0)

        # ANE-friendly softmax
        attn_max = torch.max(attn_weights, dim=-1, keepdim=True)[0]
        attn_weights = torch.exp(attn_weights - attn_max)
        attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)

        attn_output = torch.matmul(attn_weights, v)  # [B, nh, S, hd]

        # Reshape + output gate
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)  # [B, S, nh*hd]
        attn_output = attn_output * torch.sigmoid(gate)

        # Output projection (Conv2d)
        out_conv = attn_output.permute(0, 2, 1).unsqueeze(2)
        out = self.o_proj(out_conv).squeeze(2).permute(0, 2, 1)

        # KV cache (before GQA expansion)
        k_cache = k.view(B, self.num_kv_heads, self.num_groups, S, self.head_dim)[:, :, 0]
        v_cache = v.view(B, self.num_kv_heads, self.num_groups, S, self.head_dim)[:, :, 0]
        return out, k_cache, v_cache


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------

class ANETransformerLayer(nn.Module):
    def __init__(self, config: dict, layer_idx: int, is_linear: bool, max_seq_len: int = 512):
        super().__init__()
        tc = config.get("text_config", config)
        hidden = tc["hidden_size"]
        eps = tc.get("rms_norm_eps", 1e-6)

        self.is_linear = is_linear
        self.input_layernorm = ANERMSNorm(hidden, eps)
        self.post_attention_layernorm = ANERMSNorm(hidden, eps)
        self.mlp = ANEMlp(hidden, tc["intermediate_size"])

        if is_linear:
            self.attn = ANEDeltaNet(
                hidden_size=hidden,
                num_key_heads=tc["linear_num_key_heads"],
                num_value_heads=tc["linear_num_value_heads"],
                key_head_dim=tc["linear_key_head_dim"],
                value_head_dim=tc["linear_value_head_dim"],
                conv_kernel_dim=tc["linear_conv_kernel_dim"],
                eps=eps,
            )
        else:
            rope_params = tc.get("rope_parameters", {})
            self.attn = ANEFullAttention(
                hidden_size=hidden,
                num_attention_heads=tc["num_attention_heads"],
                num_key_value_heads=tc["num_key_value_heads"],
                head_dim=tc["head_dim"],
                max_seq_len=max_seq_len,
                rope_theta=rope_params.get("rope_theta", tc.get("rope_theta", 10000000.0)),
                partial_rotary_factor=rope_params.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)),
                eps=eps,
            )

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out, cache_a, cache_b = self.attn(h)
        h = residual + attn_out

        residual = h
        h = self.post_attention_layernorm(h)
        h = residual + self.mlp(h)
        return h, cache_a, cache_b


# ---------------------------------------------------------------------------
# Full prefill model
# ---------------------------------------------------------------------------

class ANEQwen35Prefill(nn.Module):
    """Complete ANE-compatible Qwen3.5 prefill model."""

    def __init__(self, config: dict, last_logit_only: bool = True, max_seq_len: int = 512):
        super().__init__()
        tc = config.get("text_config", config)
        self.config = config
        self.last_logit_only = last_logit_only

        hidden = tc["hidden_size"]
        vocab = tc["vocab_size"]
        num_layers = tc["num_hidden_layers"]
        interval = tc.get("full_attention_interval", 4)

        self.embed_tokens = nn.Embedding(vocab, hidden)

        self.layers = nn.ModuleList()
        self._layer_is_linear = []
        for i in range(num_layers):
            is_linear = (i + 1) % interval != 0
            self._layer_is_linear.append(is_linear)
            self.layers.append(ANETransformerLayer(config, i, is_linear, max_seq_len))

        self.norm = ANERMSNorm(hidden, tc.get("rms_norm_eps", 1e-6))
        self.lm_head = nn.Conv2d(hidden, vocab, 1, bias=False)
        self.num_layers = num_layers

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        h = self.embed_tokens(input_ids)

        caches = []
        for layer in self.layers:
            h, cache_a, cache_b = layer(h)
            caches.append(cache_a)
            caches.append(cache_b)

        h = self.norm(h)
        if self.last_logit_only:
            h = h[:, -1:, :]

        h_conv = h.permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(h_conv).squeeze(2).permute(0, 2, 1)

        return (logits, *caches)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_hf_weights(self, hf_state_dict: dict) -> None:
        """Load weights from HF state_dict with name mapping and reshaping."""
        own_state = self.state_dict()
        loaded = 0
        skipped = []

        for hf_key, hf_val in hf_state_dict.items():
            ane_key = self._map_key(hf_key)
            if ane_key is None:
                skipped.append(hf_key)
                continue
            if ane_key not in own_state:
                skipped.append(f"{hf_key} -> {ane_key} (not in model)")
                continue

            target_shape = own_state[ane_key].shape
            val = hf_val

            # Reshape 2D [out, in] → 4D [out, in, 1, 1] for Conv2d
            if len(target_shape) == 4 and len(val.shape) == 2:
                val = val.view(val.shape[0], val.shape[1], 1, 1)

            if val.shape != target_shape:
                skipped.append(
                    f"{hf_key} -> {ane_key}: shape {val.shape} vs {target_shape}"
                )
                continue

            own_state[ane_key] = val.to(own_state[ane_key].dtype)
            loaded += 1

        self.load_state_dict(own_state)
        print(f"Loaded {loaded}/{loaded + len(skipped)} weights")
        if skipped:
            for s in skipped[:5]:
                print(f"  SKIP: {s}")
            if len(skipped) > 5:
                print(f"  ... and {len(skipped) - 5} more")

    def _map_key(self, hf_key: str) -> str | None:
        """Map HF key to ANE model key."""
        return _map_hf_key_to_ane(hf_key)


def _map_hf_key_to_ane(hf_key: str, layer_offset: int = 0) -> str | None:
    """Map HF weight key to ANE model key.

    Parameters
    ----------
    hf_key : str
        Key from HuggingFace state_dict.
    layer_offset : int
        Subtracted from the HF layer index to produce the local layer index.
        For non-chunked models this is 0; for chunks it equals ``start_layer``.
    """
    key = hf_key
    for prefix in ["model.model.", "model."]:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break

    if key == "embed_tokens.weight":
        return "embed_tokens.weight"
    if key == "lm_head.weight":
        return "lm_head.weight"
    if key == "norm.weight":
        return "norm.weight"

    if not key.startswith("layers."):
        return None

    parts = key.split(".", 2)
    global_idx = int(parts[1])
    rest = parts[2]
    local_idx = global_idx - layer_offset
    p = f"layers.{local_idx}"

    # Layer norms + MLP
    simple = {
        "input_layernorm.weight": f"{p}.input_layernorm.weight",
        "post_attention_layernorm.weight": f"{p}.post_attention_layernorm.weight",
        "mlp.gate_proj.weight": f"{p}.mlp.gate_proj.weight",
        "mlp.up_proj.weight": f"{p}.mlp.up_proj.weight",
        "mlp.down_proj.weight": f"{p}.mlp.down_proj.weight",
    }
    if rest in simple:
        return simple[rest]

    # DeltaNet (linear attention)
    delta_map = {
        "linear_attn.in_proj_qkv.weight": f"{p}.attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight": f"{p}.attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight": f"{p}.attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight": f"{p}.attn.in_proj_b.weight",
        "linear_attn.conv1d.weight": f"{p}.attn.conv1d.weight",
        "linear_attn.A_log": f"{p}.attn.A_log",
        "linear_attn.dt_bias": f"{p}.attn.dt_bias",
        "linear_attn.norm.weight": f"{p}.attn.norm.weight",
        "linear_attn.out_proj.weight": f"{p}.attn.out_proj.weight",
    }
    if rest in delta_map:
        return delta_map[rest]

    # Full attention
    attn_map = {
        "self_attn.q_proj.weight": f"{p}.attn.q_proj.weight",
        "self_attn.q_proj.bias": f"{p}.attn.q_proj.bias",
        "self_attn.k_proj.weight": f"{p}.attn.k_proj.weight",
        "self_attn.k_proj.bias": f"{p}.attn.k_proj.bias",
        "self_attn.v_proj.weight": f"{p}.attn.v_proj.weight",
        "self_attn.v_proj.bias": f"{p}.attn.v_proj.bias",
        "self_attn.o_proj.weight": f"{p}.attn.o_proj.weight",
        "self_attn.q_norm.weight": f"{p}.attn.q_norm.weight",
        "self_attn.k_norm.weight": f"{p}.attn.k_norm.weight",
    }
    if rest in attn_map:
        return attn_map[rest]

    return None


# ---------------------------------------------------------------------------
# Chunked prefill model (for faster conversion of large seq_len)
# ---------------------------------------------------------------------------

class ANEQwen35PrefillChunk(nn.Module):
    """A chunk of the ANE-compatible Qwen3.5 prefill model.

    Splits the full model into ``num_chunks`` pieces, each containing a
    contiguous range of transformer layers.  Only the first chunk includes
    ``embed_tokens``; only the last includes ``norm`` and ``lm_head``.

    Intermediate chunks accept/return ``hidden_states`` instead of
    ``input_ids``/``logits``.  All cache outputs use *global* layer indices
    so ``cache_bridge`` works without modification.
    """

    def __init__(
        self,
        config: dict,
        chunk_idx: int,
        num_chunks: int,
        last_logit_only: bool = True,
        max_seq_len: int = 512,
    ):
        super().__init__()
        tc = config.get("text_config", config)
        self.config = config
        self.last_logit_only = last_logit_only
        self.chunk_idx = chunk_idx
        self.num_chunks = num_chunks

        hidden = tc["hidden_size"]
        vocab = tc["vocab_size"]
        num_layers = tc["num_hidden_layers"]
        interval = tc.get("full_attention_interval", 4)

        # Compute layer range for this chunk
        base, extra = divmod(num_layers, num_chunks)
        self.start_layer = sum(base + (1 if i < extra else 0) for i in range(chunk_idx))
        chunk_size = base + (1 if chunk_idx < extra else 0)
        self.end_layer = self.start_layer + chunk_size

        self.is_first = chunk_idx == 0
        self.is_last = chunk_idx == num_chunks - 1

        # Only first chunk has embedding
        if self.is_first:
            self.embed_tokens = nn.Embedding(vocab, hidden)

        # Build only this chunk's layers
        self.layers = nn.ModuleList()
        self._layer_is_linear = []
        self._global_layer_indices = list(range(self.start_layer, self.end_layer))
        for i in self._global_layer_indices:
            is_linear = (i + 1) % interval != 0
            self._layer_is_linear.append(is_linear)
            self.layers.append(ANETransformerLayer(config, i, is_linear, max_seq_len))

        # Only last chunk has norm + lm_head
        if self.is_last:
            self.norm = ANERMSNorm(hidden, tc.get("rms_norm_eps", 1e-6))
            self.lm_head = nn.Conv2d(hidden, vocab, 1, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        if self.is_first:
            h = self.embed_tokens(x)
        else:
            h = x

        caches = []
        for layer in self.layers:
            h, cache_a, cache_b = layer(h)
            caches.append(cache_a)
            caches.append(cache_b)

        if self.is_last:
            h = self.norm(h)
            if self.last_logit_only:
                h = h[:, -1:, :]
            h_conv = h.permute(0, 2, 1).unsqueeze(2)
            logits = self.lm_head(h_conv).squeeze(2).permute(0, 2, 1)
            return (logits, *caches)
        else:
            return (h, *caches)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_hf_weights(self, hf_state_dict: dict) -> None:
        """Load weights from HF state_dict, mapping global layer indices to local."""
        own_state = self.state_dict()
        loaded = 0
        skipped = []

        for hf_key, hf_val in hf_state_dict.items():
            # Filter: only process keys belonging to this chunk
            stripped = hf_key
            for prefix in ["model.model.", "model."]:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
                    break

            if stripped.startswith("layers."):
                parts = stripped.split(".", 2)
                global_idx = int(parts[1])
                if global_idx < self.start_layer or global_idx >= self.end_layer:
                    continue
            elif stripped == "embed_tokens.weight" and not self.is_first:
                continue
            elif stripped in ("lm_head.weight", "norm.weight") and not self.is_last:
                continue

            ane_key = _map_hf_key_to_ane(hf_key, layer_offset=self.start_layer)
            if ane_key is None:
                skipped.append(hf_key)
                continue
            if ane_key not in own_state:
                skipped.append(f"{hf_key} -> {ane_key} (not in model)")
                continue

            target_shape = own_state[ane_key].shape
            val = hf_val

            # Reshape 2D [out, in] → 4D [out, in, 1, 1] for Conv2d
            if len(target_shape) == 4 and len(val.shape) == 2:
                val = val.view(val.shape[0], val.shape[1], 1, 1)

            if val.shape != target_shape:
                skipped.append(
                    f"{hf_key} -> {ane_key}: shape {val.shape} vs {target_shape}"
                )
                continue

            own_state[ane_key] = val.to(own_state[ane_key].dtype)
            loaded += 1

        self.load_state_dict(own_state)
        print(f"Chunk {self.chunk_idx} (layers {self.start_layer}-{self.end_layer - 1}): "
              f"loaded {loaded}/{loaded + len(skipped)} weights")
        if skipped:
            for s in skipped[:5]:
                print(f"  SKIP: {s}")
            if len(skipped) > 5:
                print(f"  ... and {len(skipped) - 5} more")
