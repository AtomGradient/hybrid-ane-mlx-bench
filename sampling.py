"""
采样策略 — 支持 temperature, top_p, top_k
"""
from typing import Optional

import mlx.core as mx


def sample(
    logits: mx.array,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
) -> int:
    """从 logits 采样下一个 token。

    Args:
        logits: 形状 [vocab_size] 的 1-D 数组
        temperature: 采样温度，0 表示 greedy
        top_p: nucleus sampling 阈值
        top_k: top-k 采样，0 表示不限制

    Returns:
        采样得到的 token id
    """
    if temperature <= 0:
        return mx.argmax(logits).item()

    logits = logits / temperature

    if top_k > 0:
        top_k_vals = mx.topk(logits, k=min(top_k, logits.shape[-1]))
        threshold = top_k_vals[-1]
        logits = mx.where(logits < threshold, mx.array(float("-inf")), logits)

    if top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        sorted_indices = mx.argsort(probs, axis=-1)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = mx.cumsum(sorted_probs, axis=-1)

        cutoff_mask = cumsum - sorted_probs > top_p
        sorted_probs = mx.where(cutoff_mask, mx.zeros_like(sorted_probs), sorted_probs)

        sorted_probs = sorted_probs / sorted_probs.sum()
        token_idx = mx.random.categorical(mx.log(sorted_probs + 1e-10))
        return sorted_indices[token_idx].item()

    return mx.random.categorical(logits).item()
