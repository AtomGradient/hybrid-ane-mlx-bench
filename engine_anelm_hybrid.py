"""
ANE-LM Hybrid Inference Engine
================================
ANE-LM (private AppleNeuralEngine.framework) Prefill + MLX GPU Decode

Pipeline:
  1. Tokenize prompt with chat template (via mlx_lm tokenizer)
  2. Run `ane-lm prefill` subprocess: prefills tokens, exports binary cache,
     prints JSON stats to stdout.
  3. Load binary cache via anelm_cache_bridge → MLX KVCache/ArraysCache
  4. Run MLX GPU decode loop using mlx_lm's model

Usage:
    from engine_anelm_hybrid import ANELMHybridEngine

    engine = ANELMHybridEngine(
        model_path="../Qwen3.5-0.8B",
        ane_lm_bin="ANE-LM/build/ane-lm",
    )
    result = engine.generate("Hello, how are you?", max_tokens=200)
    print(result["text"])
    print(f"TTFT: {result['ttft_ms']:.0f}ms  decode: {result['decode_tps']:.1f}tok/s")
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx

from mlx_decode.anelm_cache_bridge import load_anelm_cache
from mlx_decode.mlx_model import load_mlx_model, get_language_model


def _sample_greedy(logits: mx.array) -> int:
    return int(mx.argmax(logits[0, -1, :]).item())


def _sample_temperature(logits: mx.array, temperature: float) -> int:
    scaled = logits[0, -1, :] / temperature
    probs = mx.softmax(scaled, axis=-1)
    return int(mx.random.categorical(mx.log(probs)).item())


class ANELMHybridEngine:
    """ANE-LM prefill + MLX GPU decode hybrid engine for Qwen3.5.

    Parameters
    ----------
    model_path : str
        HuggingFace model directory (Qwen3.5-0.8B etc.).
    ane_lm_bin : str
        Path to the compiled `ane-lm` binary.
    enable_thinking : bool
        Pass --enable-thinking to ane-lm (Qwen3 thinking mode).
    ane_cache : bool
        Whether to use ANE compile cache (pass --no-ane-cache to disable).
    """

    def __init__(
        self,
        model_path: str,
        ane_lm_bin: str,
        enable_thinking: bool = False,
        ane_cache: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.ane_lm_bin = Path(ane_lm_bin)
        self.enable_thinking = enable_thinking
        self.ane_cache = ane_cache

        self.config: Dict[str, Any] = json.loads(
            (self.model_path / "config.json").read_text()
        )

        print(f"Loading MLX model from {self.model_path.name} ...")
        t0 = time.time()
        self.vlm_model, self.processor = load_mlx_model(str(self.model_path))
        self.language_model = get_language_model(self.vlm_model)
        self.tokenizer = self.processor.tokenizer
        text_cfg = self.config.get("text_config", self.config)
        self.eos_token_id: int = text_cfg.get("eos_token_id", 248044)
        print(f"  MLX model loaded in {time.time() - t0:.1f}s")

        if not self.ane_lm_bin.exists():
            raise FileNotFoundError(f"ane-lm binary not found: {self.ane_lm_bin}")

    def _run_prefill(self, prompt: str, cache_path: str) -> Dict[str, Any]:
        """Run `ane-lm prefill` and return JSON stats from stdout."""
        cmd = [
            str(self.ane_lm_bin), "prefill",
            "--model", str(self.model_path),
            "--prompt", prompt,
            "--output", cache_path,
        ]
        if self.enable_thinking:
            cmd.append("--enable-thinking")
        if not self.ane_cache:
            cmd.append("--no-ane-cache")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"ane-lm prefill failed (rc={result.returncode}):\n{result.stderr}"
            )
        return json.loads(result.stdout.strip())

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Run the full ANE-LM prefill + MLX decode pipeline.

        Args:
            prompt: Raw user text (chat template applied automatically by ane-lm).
            max_tokens: Maximum tokens to generate in decode phase.
            temperature: Sampling temperature (0.0 = greedy).

        Returns:
            dict with keys:
                text          -- generated text (decoded)
                prompt_tokens -- number of prompt tokens processed
                prefill_ms    -- ANE-LM prefill wall time (ms)
                prefill_tps   -- ANE-LM prefill throughput (tok/s)
                ttft_ms       -- time-to-first-token (= prefill_ms)
                decode_tokens -- number of tokens generated
                decode_ms     -- decode wall time (ms)
                decode_tps    -- decode throughput (tok/s)
                next_token    -- first token sampled by ane-lm after prefill
        """
        with tempfile.NamedTemporaryFile(suffix=".anelm.bin", delete=False) as tmp:
            cache_path = tmp.name

        try:
            # --- Phase 1: ANE-LM prefill ---
            t_prefill_start = time.perf_counter()
            stats = self._run_prefill(prompt, cache_path)
            t_prefill_end = time.perf_counter()
            prefill_ms = (t_prefill_end - t_prefill_start) * 1000.0

            prompt_tokens: int = stats["prompt_tokens"]
            next_token: int    = stats["next_token"]
            # Use ane-lm's internal timing (more accurate, excludes subprocess overhead)
            ane_prefill_ms: float = stats["prefill_ms"]
            ane_prefill_tps: float = stats["prompt_tps"]

            # --- Phase 2: Load cache ---
            caches, _ = load_anelm_cache(cache_path, self.config)

        finally:
            try:
                os.unlink(cache_path)
            except OSError:
                pass

        # --- Phase 3: MLX decode ---
        generated_ids: List[int] = []
        t_decode_start = time.perf_counter()

        current_token = next_token
        for _ in range(max_tokens):
            if current_token == self.eos_token_id:
                break
            generated_ids.append(current_token)

            x = mx.array([[current_token]])
            out = self.language_model(x, cache=caches)
            logits = out.logits
            mx.eval(logits)

            if temperature == 0.0 or temperature < 1e-6:
                current_token = _sample_greedy(logits)
            else:
                current_token = _sample_temperature(logits, temperature)

        t_decode_end = time.perf_counter()
        decode_ms = (t_decode_end - t_decode_start) * 1000.0
        decode_tps = len(generated_ids) / (decode_ms / 1000.0) if decode_ms > 0 else 0.0

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "prefill_ms": ane_prefill_ms,
            "prefill_tps": ane_prefill_tps,
            "ttft_ms": ane_prefill_ms,
            "decode_tokens": len(generated_ids),
            "decode_ms": decode_ms,
            "decode_tps": decode_tps,
            "next_token": next_token,
        }
