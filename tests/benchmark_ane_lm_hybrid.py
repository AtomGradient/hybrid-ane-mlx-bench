#!/usr/bin/env python3
"""
ANE-LM Hybrid benchmark: ANE-LM Private API Prefill + MLX GPU Decode.

This is an end-to-end real pipeline:
  - Prefill: `ane-lm prefill` subprocess using private AppleNeuralEngine.framework
  - Cache bridge: binary cache file converted to MLX KVCache/ArraysCache format
  - Decode: MLX GPU decode loop

Measured metrics:
  - TTFT  = ANE-LM prefill time (internal measurement, from ane-lm JSON output)
  - Decode = MLX GPU decode speed (measured live)

Usage:
    python tests/benchmark_ane_lm_hybrid.py --runs 4 --append
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANE_LM_BIN = ROOT / "ANE-LM" / "build" / "ane-lm"

PROMPTS = {
    "short": "用一句话解释什么是神经网络。",
    "medium": (
        "Apple Silicon 的统一内存架构（Unified Memory Architecture）是苹果自研芯片的核心设计之一。"
        "与传统架构不同，CPU、GPU 和神经网络引擎（ANE）共享同一块物理内存池，"
        "无需在不同处理单元之间复制数据。这种设计显著降低了内存带宽消耗和延迟，"
        "使得大型语言模型的推理可以高效地在多个计算单元之间调度。"
        "在 M1 到 M4 世代的芯片中，ANE 是一个独立的协处理器，"
        "专门处理卷积和矩阵运算，峰值算力约为 11-38 TOPS。"
        "开发者通常通过 CoreML 框架间接访问 ANE，由系统自动决定算子的调度策略。"
    ),
    "long": (
        "Large language model inference on edge devices presents unique challenges that differ "
        "fundamentally from datacenter deployments. While server-side inference benefits from "
        "high-bandwidth memory (HBM), NVLink interconnects, and virtually unlimited power budgets, "
        "edge inference must operate within strict memory, power, and thermal constraints.\n\n"
        "Apple Silicon represents a compelling platform for edge LLM inference due to its unified "
        "memory architecture, which eliminates PCIe bandwidth bottlenecks between CPU and GPU memory. "
        "The Neural Engine (ANE), present in all Apple Silicon chips since M1, provides dedicated "
        "matrix multiplication hardware optimized for fixed-shape computations typical of transformer "
        "prefill phases.\n\n"
        "However, existing inference frameworks make suboptimal use of Apple Silicon's heterogeneous "
        "compute resources. MLX, Apple's open-source array framework, excels at GPU-based decode "
        "with its dynamic computation graph and lazy evaluation model. CoreML, Apple's deployment "
        "framework, achieves superior prefill throughput by routing compute-intensive attention "
        "operations to the ANE. Neither framework alone fully exploits the available hardware.\n\n"
        "We propose a disaggregated inference approach that assigns each phase to its optimal "
        "hardware: the prefill phase runs on CoreML with CPU_AND_NE compute units, leveraging the "
        "ANE for bulk matrix operations; the decode phase runs on MLX via Metal GPU acceleration, "
        "benefiting from dynamic shapes and lower per-step scheduling overhead.\n\n"
        "The key engineering challenge lies in the KV cache bridge between these two frameworks. "
        "CoreML outputs cache tensors as named NumPy arrays, while MLX-LM expects cache objects "
        "with specific offset semantics for positional encoding. For the Qwen3.5 architecture, "
        "this is further complicated by its hybrid attention design: only one in every four layers "
        "uses full self-attention with a conventional KV cache; the remaining layers use linear "
        "attention (DeltaNet) with a different recurrent state format.\n\n"
        "This paper characterizes the performance profile of our hybrid pipeline across four "
        "Qwen3.5 model variants on M1 Max hardware, measuring time-to-first-token (TTFT), "
        "decode throughput, and peak memory utilization under varying prompt lengths."
    ),
}


def benchmark_prompt(engine, label: str, prompt: str, num_runs: int = 4) -> dict | None:
    """Run the real hybrid pipeline multiple times, return median stats."""
    from engine_anelm_hybrid import ANELMHybridEngine

    print(f"\n[{label}] {num_runs} runs (run 1 is warmup)")
    results = []
    for i in range(num_runs):
        tag = " (warmup)" if i == 0 else ""
        sys.stdout.write(f"  Run {i+1}{tag}... ")
        sys.stdout.flush()
        try:
            r = engine.generate(prompt, max_tokens=200, temperature=0.0)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        print(
            f"prompt={r['prompt_tokens']}tok  "
            f"ttft={r['ttft_ms']:.0f}ms  "
            f"decode={r['decode_tps']:.1f}tok/s ({r['decode_tokens']}tok)"
        )
        results.append(r)

    if len(results) < 2:
        print(f"  [!] Not enough successful runs for {label}")
        return None

    valid = results[1:]  # drop warmup
    med_ttft   = statistics.median(r["ttft_ms"]   for r in valid)
    med_ptps   = statistics.median(r["prefill_tps"] for r in valid)
    med_decode = statistics.median(r["decode_tps"] for r in valid)
    med_dtok   = statistics.median(r["decode_tokens"] for r in valid)
    prompt_tokens = valid[0]["prompt_tokens"]

    print(
        f"  -> Median ({len(valid)} runs): "
        f"ttft={med_ttft:.0f}ms  prefill={med_ptps:.1f}tok/s  "
        f"decode={med_decode:.1f}tok/s"
    )

    return {
        "model": "Qwen3.5-0.8B",
        "quant": "FP16",
        "backend": "ane_lm_hybrid",
        "prompt_length": label,
        "status": "OK",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": round(med_dtok),
        "prefill_time_ms": round(med_ttft, 2),
        "prefill_tokens_per_sec": round(med_ptps, 1),
        "ttft_ms": round(med_ttft, 2),
        "decode_tokens_per_sec": round(med_decode, 1),
        "peak_memory_gb": "",
        "total_time_ms": "",
        "note": "prefill=ANE-LM private API (measured); decode=MLX GPU (measured)",
    }


def main():
    parser = argparse.ArgumentParser(description="ANE-LM hybrid benchmark (real pipeline)")
    parser.add_argument("--model", default=str(ROOT.parent / "Qwen3.5-0.8B"))
    parser.add_argument("--ane-lm-bin", default=str(ANE_LM_BIN))
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--prompt-lengths", nargs="+",
                        default=["short", "medium", "long"],
                        choices=["short", "medium", "long"])
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    ane_bin = Path(args.ane_lm_bin)
    if not ane_bin.exists():
        sys.exit(f"Error: ane-lm binary not found: {ane_bin}")

    print("ANE-LM Hybrid Benchmark (REAL end-to-end pipeline)")
    print(f"Model:   {args.model}")
    print(f"ANE-LM:  {ane_bin}")
    print(f"Prompts: {args.prompt_lengths}, runs={args.runs}")
    print()

    from engine_anelm_hybrid import ANELMHybridEngine
    engine = ANELMHybridEngine(
        model_path=args.model,
        ane_lm_bin=str(ane_bin),
        enable_thinking=args.enable_thinking,
    )

    results = []
    for label in args.prompt_lengths:
        r = benchmark_prompt(engine, label, PROMPTS[label], args.runs)
        if r:
            results.append(r)

    if not results:
        sys.exit("No results collected.")

    print("\n" + "=" * 78)
    print("ANE-LM HYBRID SUMMARY (prefill=ANE private API, decode=MLX GPU, MEASURED)")
    print("=" * 78)
    print(f"{'Prompt':<8} {'PrTok':>6} {'TTFT(ms)':>10} {'Prefill(t/s)':>13} {'Decode(t/s)':>12}")
    print("-" * 78)
    for r in results:
        print(f"{r['prompt_length']:<8} {r['prompt_tokens']:>6} {r['ttft_ms']:>10.0f} "
              f"{r['prefill_tokens_per_sec']:>13.1f} {r['decode_tokens_per_sec']:>12.1f}")

    results_path = ROOT / "results" / "benchmark_results.json"
    csv_path     = ROOT / "results" / "benchmark_results.csv"

    existing = []
    if args.append and results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        replaced = {(r["model"], r["backend"], r["prompt_length"]) for r in results}
        existing = [e for e in existing
                    if (e["model"], e["backend"], e["prompt_length"]) not in replaced]

    all_results = existing + results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    csv_keys = [k for k in all_results[0].keys() if k != "note"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved {len(results)} REAL results to {results_path}")


if __name__ == "__main__":
    main()
