"""
MLX 模型加载与单步 decode

使用 mlx_vlm 而非 mlx_lm（Qwen3.5 是 VLM）
"""
from typing import Optional, Tuple
from pathlib import Path

import mlx.core as mx


def load_mlx_model(model_path: str):
    """加载 mlx_vlm 模型。

    Args:
        model_path: HuggingFace 模型目录

    Returns:
        (model, processor) tuple
    """
    from mlx_vlm import load

    model, processor = load(model_path)
    return model, processor


def get_language_model(model):
    """从 VLM 中提取语言模型部分。"""
    if hasattr(model, "language_model"):
        return model.language_model
    return model


def decode_step(
    language_model,
    token_id: int,
    cache: list,
) -> Tuple[mx.array, list]:
    """执行单步 decode。

    Args:
        language_model: mlx_vlm 的 LanguageModel
        token_id: 当前 token ID
        cache: 混合 cache 列表 (KVCache + ArraysCache)

    Returns:
        (logits [1, 1, vocab_size], updated_cache)
    """
    x = mx.array([[token_id]])
    out = language_model(x, cache=cache)
    logits = out.logits
    mx.eval(logits)
    return logits, cache


def generate_tokens(
    language_model,
    first_token: int,
    cache: list,
    max_tokens: int = 200,
    eos_token_id: int = 248044,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list:
    """使用 MLX decode 生成 token 序列。

    Args:
        language_model: mlx_vlm LanguageModel
        first_token: prefill 产生的首个 token
        cache: 混合 cache（从 CoreML prefill 桥接而来）
        max_tokens: 最大生成长度
        eos_token_id: 结束 token
        temperature: 采样温度
        top_p: nucleus sampling

    Returns:
        生成的 token ID 列表
    """
    from ..sampling import sample

    generated = []
    next_token = first_token

    for _ in range(max_tokens):
        if next_token == eos_token_id:
            break
        generated.append(next_token)

        logits, cache = decode_step(language_model, next_token, cache)
        next_logits = logits[0, -1, :]
        next_token = sample(next_logits, temperature=temperature, top_p=top_p)

    return generated
