#!/usr/bin/env python3
"""
ANE-LM benchmark: measures TTFT and decode throughput using the ANE-LM binary.

ANE-LM uses private Apple Neural Engine APIs for matrix operations.
Both prefill and decode run on ANE (sequential, one token at a time).

Usage:
    python tests/benchmark_ane_lm.py --runs 4 --append
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANE_LM_BIN = ROOT / "ANE-LM" / "build" / "ane-lm"

# Same prompts as benchmark.py for consistency
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


def parse_output(stderr: str) -> dict:
    """Parse ANE-LM stderr output to extract timing metrics."""
    result = {}
    m = re.search(r"Prompt:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec", stderr)
    if m:
        result["prompt_tokens"] = int(m.group(1))
        result["prompt_tps"] = float(m.group(2))
        result["ttft_ms"] = result["prompt_tokens"] / result["prompt_tps"] * 1000.0
        result["prefill_time_ms"] = result["ttft_ms"]
    m = re.search(r"Generation:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec", stderr)
    if m:
        result["generated_tokens"] = int(m.group(1))
        result["decode_tps"] = float(m.group(2))
    return result


def run_once(model_path: str, prompt: str, max_tokens: int = 200) -> dict:
    """Run ANE-LM once and return parsed metrics."""
    cmd = [
        str(ANE_LM_BIN), "generate",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0",
        "--repeat-penalty", "1.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:300]}", file=sys.stderr)
        return {}
    return parse_output(result.stderr)


def benchmark_prompt(model_path: str, label: str, prompt: str,
                     num_runs: int = 4, max_tokens: int = 200) -> dict | None:
    """Run multiple times; discard first (warmup), report median of rest."""
    print(f"\n[{label}] {num_runs} runs, run 1 is warmup")
    all_results = []
    for i in range(num_runs):
        tag = " (warmup)" if i == 0 else ""
        sys.stdout.write(f"  Run {i+1}{tag}... ")
        sys.stdout.flush()
        m = run_once(model_path, prompt, max_tokens)
        if not m:
            print("FAILED")
            continue
        ttft = m.get("ttft_ms", 0)
        dtps = m.get("decode_tps", 0)
        print(f"prompt={m.get('prompt_tokens','?')}tok  ttft={ttft:.0f}ms  decode={dtps:.1f}tok/s")
        all_results.append(m)

    if len(all_results) < 2:
        return None

    valid = all_results[1:]  # drop warmup
    med_ttft = statistics.median(r["ttft_ms"] for r in valid)
    med_dtps = statistics.median(r["decode_tps"] for r in valid)
    med_ptps = statistics.median(r["prompt_tps"] for r in valid)
    med_gtok = statistics.median(r["generated_tokens"] for r in valid)
    prompt_tokens = valid[0]["prompt_tokens"]

    print(f"  -> Median ({len(valid)} runs): ttft={med_ttft:.0f}ms  decode={med_dtps:.1f}tok/s  "
          f"prefill={med_ptps:.1f}tok/s  prompt_tokens={prompt_tokens}")
    return {
        "model": "Qwen3.5-0.8B",
        "quant": "FP16",
        "backend": "ane_lm",
        "prompt_length": label,
        "status": "OK",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(med_gtok),
        "prefill_time_ms": round(med_ttft, 2),
        "prefill_tokens_per_sec": round(med_ptps, 1),
        "ttft_ms": round(med_ttft, 2),
        "decode_tokens_per_sec": round(med_dtps, 1),
        "peak_memory_gb": "",
        "total_time_ms": "",
    }


def main():
    parser = argparse.ArgumentParser(description="ANE-LM benchmark")
    parser.add_argument("--model", default=str(ROOT.parent / "Qwen3.5-0.8B"))
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--prompt-lengths", nargs="+",
                        default=["short", "medium", "long"],
                        choices=["short", "medium", "long"])
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if not ANE_LM_BIN.exists():
        sys.exit(f"Error: {ANE_LM_BIN} not found. Build ANE-LM first.")

    print(f"ANE-LM: {ANE_LM_BIN}")
    print(f"Model:  {args.model}")
    print(f"Prompts: {args.prompt_lengths}, runs={args.runs}")

    results = []
    for label in args.prompt_lengths:
        r = benchmark_prompt(args.model, label, PROMPTS[label], args.runs, args.max_tokens)
        if r:
            results.append(r)

    if not results:
        sys.exit("No results.")

    print("\n" + "=" * 78)
    print("ANE-LM BENCHMARK SUMMARY (M2 Ultra 192GB)")
    print("=" * 78)
    print(f"{'Backend':<10} {'Prompt':<8} {'PrTok':>6} {'TTFT(ms)':>10} {'Prefill(t/s)':>13} {'Decode(t/s)':>12}")
    print("-" * 78)
    for r in results:
        print(f"{r['backend']:<10} {r['prompt_length']:<8} {r['prompt_tokens']:>6} "
              f"{r['ttft_ms']:>10.0f} {r['prefill_tokens_per_sec']:>13.1f} "
              f"{r['decode_tokens_per_sec']:>12.1f}")

    results_path = ROOT / "results" / "benchmark_results.json"
    csv_path = ROOT / "results" / "benchmark_results.csv"

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

    keys = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved {len(results)} new results to {results_path}")


if __name__ == "__main__":
    main()
