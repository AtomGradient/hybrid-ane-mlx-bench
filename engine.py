"""
Qwen3.5 混合推理引擎 — CoreML (ANE) Prefill + MLX (GPU) Decode

使用方式:
    from engine import HybridInferenceEngine

    engine = HybridInferenceEngine("../Qwen3.5-0.8B")
    result = engine.generate("你好，请介绍苹果神经网络引擎")
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_decode.cache_bridge import coreml_to_mlx_cache
from mlx_decode.mlx_model import load_mlx_model, get_language_model, decode_step
from sampling import sample


class HybridInferenceEngine:
    """混合推理引擎：ANE prefill + GPU decode。

    Parameters
    ----------
    model_path : str
        HuggingFace 模型目录路径（如 ``../Qwen3.5-0.8B``）。
    coreml_path : str, optional
        CoreML .mlpackage 路径（单体模型），或包含 chunk 文件的目录。
        如果为 None 则跳过 CoreML，用纯 MLX 做 prefill（用于开发调试）。
        支持自动发现 chunked 模型：若单体文件不存在，搜索
        ``*_chunk*of*.mlpackage`` 文件。
    """

    def __init__(
        self,
        model_path: str,
        coreml_path: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config = json.loads((self.model_path / "config.json").read_text())
        self.text_config = self.config.get("text_config", self.config)

        model_name = self.model_path.name
        print(f"Loading {model_name} ...")

        # MLX model (for decode, and optionally for prefill fallback)
        t0 = time.time()
        self.vlm_model, self.processor = load_mlx_model(str(self.model_path))
        self.language_model = get_language_model(self.vlm_model)
        print(f"  MLX model loaded in {time.time() - t0:.1f}s")

        # CoreML model(s) (optional — for ANE prefill)
        self.coreml_model = None
        self.coreml_chunks: list = []  # list of MLModel when using chunked prefill
        self._load_coreml(coreml_path)

        # Tokenizer
        self.tokenizer = self.processor.tokenizer
        self.eos_token_id = self.text_config.get("eos_token_id", 248044)

        coreml_status = "OFF"
        if self.coreml_chunks:
            coreml_status = f"ON ({len(self.coreml_chunks)} chunks)"
        elif self.coreml_model:
            coreml_status = "ON"
        print(f"  {model_name} ready (coreml={coreml_status})")

    def _load_coreml(self, coreml_path: Optional[str]) -> None:
        """Load single or chunked CoreML model(s)."""
        if not coreml_path:
            return

        path = Path(coreml_path)

        # Case 1: direct path to a single .mlpackage
        if path.exists() and path.suffix == ".mlpackage":
            import coremltools as ct
            t0 = time.time()
            self.coreml_model = ct.models.MLModel(
                str(path),
                compute_units=ct.ComputeUnit.ALL,
            )
            print(f"  CoreML model loaded in {time.time() - t0:.1f}s")
            return

        # Case 2: path doesn't exist as-is — try chunk discovery
        # Look in the parent directory for chunk files matching the pattern
        search_dir = path.parent if not path.is_dir() else path
        if not search_dir.exists():
            return

        # Derive the base pattern from coreml_path
        # e.g. "models/ane_qwen3_5-0_8b_prefill_seq512.mlpackage"
        #   -> search for "ane_qwen3_5-0_8b_prefill_seq512_chunk*of*.mlpackage"
        if path.suffix == ".mlpackage":
            stem = path.stem  # e.g. "ane_qwen3_5-0_8b_prefill_seq512"
            chunk_pattern = f"{stem}_chunk*of*.mlpackage"
        else:
            # coreml_path is a directory — search for any chunk files
            chunk_pattern = "*_chunk*of*.mlpackage"

        chunk_files = sorted(search_dir.glob(chunk_pattern))
        if not chunk_files:
            # Also try: the path itself might be a directory with chunks
            if path.is_dir():
                chunk_files = sorted(path.glob("*_chunk*of*.mlpackage"))
            if not chunk_files:
                return

        import coremltools as ct
        t0 = time.time()
        for cf in chunk_files:
            model = ct.models.MLModel(
                str(cf),
                compute_units=ct.ComputeUnit.ALL,
            )
            self.coreml_chunks.append(model)
            print(f"  Loaded chunk: {cf.name}")
        print(f"  {len(self.coreml_chunks)} CoreML chunks loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _prefill_mlx(
        self,
        input_ids: list[int],
    ) -> tuple[mx.array, list]:
        """纯 MLX prefill（fallback，用于 CoreML 不可用时）。"""
        from mlx_vlm.models.cache import make_prompt_cache

        cache = self.language_model.make_cache()
        x = mx.array([input_ids])

        out = self.language_model(x, cache=cache)
        logits = out.logits
        mx.eval(logits)

        last_logits = logits[0, -1, :]
        return last_logits, cache

    def _prefill_coreml(
        self,
        input_ids: list[int],
        seq_len: int,
    ) -> tuple[mx.array, list]:
        """CoreML (ANE) prefill — single model."""
        # Pad input to fixed seq_len
        padded = [0] * (seq_len - len(input_ids)) + input_ids
        mask = [0] * (seq_len - len(input_ids)) + [1] * len(input_ids)

        coreml_input = {
            "input_ids": np.array([padded], dtype=np.int32),
        }
        # 仅当模型接受 attention_mask 时才传入
        try:
            model_input_names = {
                inp.name for inp in self.coreml_model._spec.description.input
            }
        except Exception:
            model_input_names = set()
        if "attention_mask" in model_input_names:
            coreml_input["attention_mask"] = np.array([mask], dtype=np.int32)

        coreml_out = self.coreml_model.predict(coreml_input)

        # Extract logits for last real token position
        logits_np = coreml_out["logits"]  # [1, logits_seq, vocab_size]
        if logits_np.shape[1] == 1:
            last_logits = mx.array(logits_np[0, 0, :])
        else:
            last_logits = mx.array(logits_np[0, len(input_ids) - 1, :])

        # Convert CoreML cache outputs to MLX cache
        cache = coreml_to_mlx_cache(
            coreml_out,
            self.config,
            prompt_len=len(input_ids),
        )

        return last_logits, cache

    def _prefill_chunked(
        self,
        input_ids: list[int],
        seq_len: int,
    ) -> tuple[mx.array, list]:
        """CoreML (ANE) prefill — chunked models."""
        padded = [0] * (seq_len - len(input_ids)) + input_ids
        all_cache_outputs: dict = {}

        for chunk_idx, chunk_model in enumerate(self.coreml_chunks):
            if chunk_idx == 0:
                chunk_input = {
                    "input_ids": np.array([padded], dtype=np.int32),
                }
            else:
                chunk_input = {
                    "hidden_states": hidden_states,
                }

            chunk_out = chunk_model.predict(chunk_input)

            # Collect cache outputs (all keys except hidden_states/logits)
            for key, val in chunk_out.items():
                if key not in ("hidden_states", "logits"):
                    all_cache_outputs[key] = val

            # Pass hidden_states to next chunk (if not last)
            if "hidden_states" in chunk_out:
                hidden_states = chunk_out["hidden_states"]

        # Extract logits from last chunk
        logits_np = chunk_out["logits"]  # [1, logits_seq, vocab_size]
        if logits_np.shape[1] == 1:
            last_logits = mx.array(logits_np[0, 0, :])
        else:
            last_logits = mx.array(logits_np[0, len(input_ids) - 1, :])

        # Convert all collected cache outputs to MLX cache
        cache = coreml_to_mlx_cache(
            all_cache_outputs,
            self.config,
            prompt_len=len(input_ids),
        )

        return last_logits, cache

    @staticmethod
    def _select_seq_len(n: int, options: list[int] = [64, 128, 256, 512]) -> int:
        """选择最近的大于等于 n 的序列长度。"""
        for L in options:
            if n <= L:
                return L
        raise ValueError(f"输入过长 ({n} tokens)，最大支持 {options[-1]}")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        verbose: bool = True,
    ) -> str:
        """生成文本。

        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（0 = greedy）
            top_p: nucleus sampling 阈值
            verbose: 是否打印性能信息

        Returns:
            生成的文本字符串
        """
        # Tokenize
        tokens = self.tokenizer.encode(prompt)
        if verbose:
            print(f"Prompt: {len(tokens)} tokens")

        # Prefill
        t0 = time.time()
        if self.coreml_chunks:
            seq_len = self._select_seq_len(len(tokens))
            last_logits, cache = self._prefill_chunked(tokens, seq_len)
            prefill_backend = f"ANE({len(self.coreml_chunks)}chunks)"
        elif self.coreml_model:
            seq_len = self._select_seq_len(len(tokens))
            last_logits, cache = self._prefill_coreml(tokens, seq_len)
            prefill_backend = "ANE"
        else:
            last_logits, cache = self._prefill_mlx(tokens)
            prefill_backend = "MLX"

        prefill_ms = (time.time() - t0) * 1000
        if verbose:
            print(f"Prefill ({prefill_backend}): {prefill_ms:.0f}ms "
                  f"({len(tokens) / (prefill_ms / 1000):.0f} tok/s)")

        # Sample first token from prefill logits
        first_token = sample(last_logits, temperature=temperature, top_p=top_p)

        # Decode loop
        generated = []
        next_token = first_token
        t1 = time.time()

        for _ in range(max_new_tokens):
            if next_token == self.eos_token_id:
                break
            generated.append(next_token)

            logits, cache = decode_step(self.language_model, next_token, cache)
            next_logits = logits[0, -1, :]
            next_token = sample(next_logits, temperature=temperature, top_p=top_p)

        decode_time = time.time() - t1
        if verbose and generated:
            speed = len(generated) / max(decode_time, 1e-6)
            print(f"Decode (GPU): {speed:.1f} tok/s ({len(generated)} tokens)")

        return self.tokenizer.decode(generated)

    # ------------------------------------------------------------------
    # Chat convenience
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """简单的单轮对话接口。"""
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3.5 混合推理引擎")
    parser.add_argument("model_path", help="模型目录路径")
    parser.add_argument("--coreml", default=None, help="CoreML .mlpackage 路径")
    parser.add_argument("--prompt", default="你好，请简单介绍一下自己。",
                        help="输入提示")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    engine = HybridInferenceEngine(args.model_path, args.coreml)
    result = engine.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"\n--- Output ---\n{result}")


if __name__ == "__main__":
    main()
