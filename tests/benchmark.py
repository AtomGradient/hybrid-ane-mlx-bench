#!/usr/bin/env python3
"""
Qwen3.5 混合推理基准测试 — 论文数据采集

用法:
    python tests/benchmark.py --backends baseline_mlx --models 0.8B 2B-8bit 2B-bf16 9B-8bit
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlx.core as mx

# ---------------------------------------------------------------------------
# 固定 Prompt（保证可复现）
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 模型名称 → 路径映射
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "0.8B":     "../Qwen3.5-0.8B",
    "2B-8bit":  "../Qwen3.5-2B-8bit",
    "2B-bf16":  "../Qwen3.5-2B-bf16",
    "9B-8bit":  "../Qwen3.5-9B-8bit",
}

MODEL_QUANT = {
    "0.8B":     "FP16",
    "2B-8bit":  "8-bit",
    "2B-bf16":  "BF16",
    "9B-8bit":  "8-bit",
}

MAX_GENERATE_TOKENS = 200

# ---------------------------------------------------------------------------
# 单次测试运行
# ---------------------------------------------------------------------------

def run_single(engine, prompt_text: str, tokenizer) -> dict:
    """执行一次 prefill + decode，返回详细指标。"""
    from mlx_decode.mlx_model import decode_step
    from sampling import sample

    tokens = tokenizer.encode(prompt_text)
    prompt_len = len(tokens)

    eos_token_id = engine.eos_token_id

    # 清理缓存 & 重置峰值内存
    mx.clear_cache()
    mx.reset_peak_memory()
    mem_before = mx.get_active_memory()

    # 重置 mlx_vlm 的 RoPE position_ids 缓存（不同 prompt 长度需要重新计算）
    if hasattr(engine.language_model, "_position_ids"):
        engine.language_model._position_ids = None
    if hasattr(engine.language_model, "_rope_deltas"):
        engine.language_model._rope_deltas = None

    # --- Prefill ---
    t_prefill_start = time.perf_counter()
    if engine.coreml_chunks:
        seq_len = engine._select_seq_len(prompt_len)
        last_logits, cache = engine._prefill_chunked(tokens, seq_len)
    elif engine.coreml_model:
        seq_len = engine._select_seq_len(prompt_len)
        last_logits, cache = engine._prefill_coreml(tokens, seq_len)
    else:
        last_logits, cache = engine._prefill_mlx(tokens)
    mx.eval(last_logits)
    t_prefill_end = time.perf_counter()

    prefill_time_ms = (t_prefill_end - t_prefill_start) * 1000

    # --- First token ---
    first_token = sample(last_logits, temperature=0.0)  # greedy for reproducibility
    t_first_token = time.perf_counter()
    ttft_ms = (t_first_token - t_prefill_start) * 1000

    # --- Decode loop ---
    generated = []
    next_token = first_token
    t_decode_start = time.perf_counter()

    for _ in range(MAX_GENERATE_TOKENS):
        if next_token == eos_token_id:
            break
        generated.append(next_token)
        logits, cache = decode_step(engine.language_model, next_token, cache)
        next_logits = logits[0, -1, :]
        next_token = sample(next_logits, temperature=0.0)

    t_decode_end = time.perf_counter()
    decode_time = t_decode_end - t_decode_start
    total_time_ms = (t_decode_end - t_prefill_start) * 1000

    decode_tokens = len(generated)
    decode_tok_per_sec = decode_tokens / max(decode_time, 1e-6)

    peak_mem = mx.get_peak_memory()
    peak_memory_gb = peak_mem / (1024 ** 3)

    prefill_tok_per_sec = prompt_len / max(prefill_time_ms / 1000, 1e-6)

    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": decode_tokens,
        "prefill_time_ms": round(prefill_time_ms, 2),
        "prefill_tokens_per_sec": round(prefill_tok_per_sec, 1),
        "ttft_ms": round(ttft_ms, 2),
        "decode_tokens_per_sec": round(decode_tok_per_sec, 1),
        "peak_memory_gb": round(peak_memory_gb, 2),
        "total_time_ms": round(total_time_ms, 2),
    }


# ---------------------------------------------------------------------------
# 结果保存
# ---------------------------------------------------------------------------

RESULTS_DIR = ROOT / "results"
JSON_PATH = RESULTS_DIR / "benchmark_results.json"
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"

CSV_FIELDS = [
    "model", "quant", "backend", "prompt_length", "prompt_tokens",
    "generated_tokens", "prefill_time_ms", "prefill_tokens_per_sec",
    "ttft_ms", "decode_tokens_per_sec", "peak_memory_gb", "total_time_ms",
    "status",
]


def load_existing_results() -> list[dict]:
    if JSON_PATH.exists():
        return json.loads(JSON_PATH.read_text())
    return []


def save_results(results: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # JSON
    JSON_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    # CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------


# CoreML models are converted from original HF weights (FP16/BF16), not quantized.
# Map model names to the slug used in CoreML filenames.
COREML_SLUG = {
    "0.8B":     "qwen3_5-0_8b",
    "2B-8bit":  None,            # 8-bit incompatible with CoreML conversion
    "2B-bf16":  "qwen3_5-2b-bf16",
    "9B-8bit":  "qwen3_5-9b",    # converted from FP16 HF weights
}


def find_coreml_path(model_name: str, prompt_len: int) -> str | None:
    """查找模型对应的 CoreML .mlpackage 路径，选择匹配 prompt 长度的 seq_len。

    CoreML 模型保存在 models/ 目录下，命名格式：
        ane_{name_slug}_prefill_seq{N}.mlpackage          (单体)
        ane_{name_slug}_prefill_seq{N}_chunk*of*.mlpackage (分块)
    """
    from engine import HybridInferenceEngine

    slug = COREML_SLUG.get(model_name)
    if slug is None:
        return None  # model not compatible with CoreML conversion

    seq_len = HybridInferenceEngine._select_seq_len(prompt_len)
    name_slug = slug
    coreml_name = f"ane_{name_slug}_prefill_seq{seq_len}.mlpackage"
    coreml_path = ROOT / "models" / coreml_name
    if coreml_path.exists():
        return str(coreml_path)
    # 检查是否存在分块文件（engine.py 会自动发现）
    chunk_pattern = f"ane_{name_slug}_prefill_seq{seq_len}_chunk*of*.mlpackage"
    if list((ROOT / "models").glob(chunk_pattern)):
        return str(coreml_path)  # 返回虚拟路径，engine.py 自动发现 chunks
    return None


def run_benchmark(args):
    from engine import HybridInferenceEngine

    # Optional power monitoring (requires sudo asitop running)
    power_monitor = None
    if not args.no_power:
        try:
            from tests.parse_power import PowerMonitor
            power_monitor = PowerMonitor()
            if power_monitor.filepath:
                print(f"[PowerMonitor] Tracking power via {power_monitor.filepath}")
            else:
                power_monitor = None
        except Exception as e:
            print(f"[PowerMonitor] Disabled: {e}")

    results = load_existing_results()
    total_configs = len(args.models) * len(args.backends) * len(args.prompt_lengths)
    config_idx = 0

    for model_name in args.models:
        model_path = MODEL_MAP.get(model_name)
        if not model_path:
            print(f"[WARN] Unknown model: {model_name}, skipping")
            continue

        for backend in args.backends:
            if backend == "baseline_mlx":
                # baseline_mlx: 加载一次，跑所有 prompt 长度
                print(f"\n{'='*60}")
                print(f"Loading model: Qwen3.5-{model_name} (backend={backend})")
                print(f"{'='*60}")

                try:
                    engine = HybridInferenceEngine(model_path, coreml_path=None)
                except Exception as e:
                    print(f"[ERROR] Failed to load model: {e}")
                    for pl in args.prompt_lengths:
                        config_idx += 1
                        entry = {
                            "model": f"Qwen3.5-{model_name}",
                            "quant": MODEL_QUANT.get(model_name, "?"),
                            "backend": backend,
                            "prompt_length": pl,
                            "status": f"ERROR: {e}",
                        }
                        for f in CSV_FIELDS:
                            if f not in entry:
                                entry[f] = ""
                        results.append(entry)
                        save_results(results)
                    continue

                for pl in args.prompt_lengths:
                    config_idx += 1
                    results = _run_config(engine, model_name, backend, pl,
                                          args.num_runs, config_idx, total_configs, results,
                                          power_monitor=power_monitor)

                del engine
                gc.collect()
                mx.clear_cache()

            elif backend == "hybrid_ane":
                # hybrid_ane: 每个 prompt 长度需要对应 seq_len 的 CoreML 模型
                # 先加载一次 tokenizer 获取各 prompt 的 token 数
                tmp_engine = HybridInferenceEngine(model_path, coreml_path=None)
                prompt_token_counts = {
                    pl: len(tmp_engine.tokenizer.encode(PROMPTS[pl]))
                    for pl in args.prompt_lengths
                }
                del tmp_engine
                gc.collect()
                mx.clear_cache()

                for pl in args.prompt_lengths:
                    config_idx += 1
                    prompt_len = prompt_token_counts[pl]

                    coreml_path = find_coreml_path(model_name, prompt_len)
                    if not coreml_path:
                        print(f"\n[{config_idx}/{total_configs}] "
                              f"{model_name} / {backend} / {pl}: "
                              f"CONVERSION_FAILED (no CoreML for seq_len needed)")
                        entry = {
                            "model": f"Qwen3.5-{model_name}",
                            "quant": MODEL_QUANT.get(model_name, "?"),
                            "backend": backend,
                            "prompt_length": pl,
                            "status": "CONVERSION_FAILED",
                        }
                        for f in CSV_FIELDS:
                            if f not in entry:
                                entry[f] = ""
                        results.append(entry)
                        save_results(results)
                        continue

                    print(f"\n{'='*60}")
                    print(f"Loading: Qwen3.5-{model_name} + CoreML ({coreml_path})")
                    print(f"{'='*60}")

                    try:
                        engine = HybridInferenceEngine(model_path, coreml_path=coreml_path)
                    except Exception as e:
                        print(f"[ERROR] Failed to load: {e}")
                        entry = {
                            "model": f"Qwen3.5-{model_name}",
                            "quant": MODEL_QUANT.get(model_name, "?"),
                            "backend": backend,
                            "prompt_length": pl,
                            "status": f"ERROR: {e}",
                        }
                        for f in CSV_FIELDS:
                            if f not in entry:
                                entry[f] = ""
                        results.append(entry)
                        save_results(results)
                        continue

                    results = _run_config(engine, model_name, backend, pl,
                                          args.num_runs, config_idx, total_configs, results,
                                          power_monitor=power_monitor)

                    del engine
                    gc.collect()
                    mx.clear_cache()

    return results


def _run_config(engine, model_name, backend, pl, num_runs, config_idx,
                total_configs, results, power_monitor=None):
    """运行单个配置的多次测试，返回更新后的 results。"""
    prompt_text = PROMPTS[pl]
    phase_key = f"{model_name}_{backend}_{pl}"

    print(f"\n[{config_idx}/{total_configs}] "
          f"Qwen3.5-{model_name} / {backend} / {pl} "
          f"({num_runs} runs)")

    # Start power tracking for this config (covers all runs including warmup)
    if power_monitor:
        power_monitor.mark(phase_key)

    run_metrics = []
    for run_idx in range(num_runs):
        try:
            mx.clear_cache()
            gc.collect()

            metrics = run_single(engine, prompt_text, engine.tokenizer)
            run_metrics.append(metrics)

            tag = "(warmup)" if run_idx == 0 else ""
            print(f"  Run {run_idx+1}: "
                  f"prefill={metrics['prefill_time_ms']:.0f}ms "
                  f"decode={metrics['decode_tokens_per_sec']:.1f}tok/s "
                  f"ttft={metrics['ttft_ms']:.0f}ms "
                  f"mem={metrics['peak_memory_gb']:.2f}GB "
                  f"{tag}")

        except MemoryError:
            print(f"  Run {run_idx+1}: OOM!")
            run_metrics.append(None)
        except Exception as e:
            print(f"  Run {run_idx+1}: ERROR - {e}")
            traceback.print_exc()
            run_metrics.append(None)

    # 丢弃第 1 次（冷启动），取后 N-1 次的中位数
    valid_runs = [m for m in run_metrics[1:] if m is not None]

    if not valid_runs:
        status = "OOM" if any(m is None for m in run_metrics) else "ERROR"
        entry = {
            "model": f"Qwen3.5-{model_name}",
            "quant": MODEL_QUANT.get(model_name, "?"),
            "backend": backend,
            "prompt_length": pl,
            "status": status,
        }
        for f in CSV_FIELDS:
            if f not in entry:
                entry[f] = ""
    else:
        median_metrics = {}
        for key in valid_runs[0]:
            vals = [r[key] for r in valid_runs]
            median_metrics[key] = round(statistics.median(vals), 2)

        entry = {
            "model": f"Qwen3.5-{model_name}",
            "quant": MODEL_QUANT.get(model_name, "?"),
            "backend": backend,
            "prompt_length": pl,
            "status": "OK",
            **median_metrics,
        }

        print(f"  → Median ({len(valid_runs)} runs): "
              f"prefill={median_metrics['prefill_time_ms']:.0f}ms "
              f"decode={median_metrics['decode_tokens_per_sec']:.1f}tok/s "
              f"ttft={median_metrics['ttft_ms']:.0f}ms "
              f"peak_mem={median_metrics['peak_memory_gb']:.2f}GB")

    # Attach power data for this config
    if power_monitor and power_monitor._current_phase == phase_key:
        power_monitor.stop()
        phase_power = power_monitor.get_phase_data().get(phase_key, {})
        if phase_power.get("count", 0) > 0:
            entry["power_cpu_avg_mw"] = phase_power["cpu_power_mw"]["avg"]
            entry["power_gpu_avg_mw"] = phase_power["gpu_power_mw"]["avg"]
            entry["power_ane_avg_mw"] = phase_power["ane_power_mw"]["avg"]
            entry["power_combined_avg_mw"] = phase_power["combined_power_mw"]["avg"]
            entry["power_samples"] = phase_power["count"]
            print(f"  → Power: CPU={entry['power_cpu_avg_mw']:.0f}mW "
                  f"GPU={entry['power_gpu_avg_mw']:.0f}mW "
                  f"ANE={entry['power_ane_avg_mw']:.0f}mW "
                  f"({entry['power_samples']} samples)")

    results.append(entry)
    save_results(results)
    return results


def print_summary_table(results: list[dict]):
    """打印终端对比表格。"""
    print(f"\n{'='*90}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*90}")
    header = (f"{'Model':<18} {'Quant':<6} {'Backend':<15} {'Prompt':<8} "
              f"{'TTFT(ms)':>10} {'Decode(t/s)':>12} {'Mem(GB)':>9} {'Status':<10}")
    print(header)
    print("-" * 90)

    for r in results:
        ttft = r.get("ttft_ms", "")
        decode = r.get("decode_tokens_per_sec", "")
        mem = r.get("peak_memory_gb", "")
        ttft_str = f"{ttft:.0f}" if isinstance(ttft, (int, float)) else str(ttft)
        decode_str = f"{decode:.1f}" if isinstance(decode, (int, float)) else str(decode)
        mem_str = f"{mem:.2f}" if isinstance(mem, (int, float)) else str(mem)

        line = (f"{r.get('model',''):<18} {r.get('quant',''):<6} "
                f"{r.get('backend',''):<15} {r.get('prompt_length',''):<8} "
                f"{ttft_str:>10} {decode_str:>12} {mem_str:>9} "
                f"{r.get('status',''):<10}")
        print(line)

    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 混合推理基准测试（论文数据采集）"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["0.8B", "2B-8bit", "2B-bf16", "9B-8bit"],
        choices=["0.8B", "2B-8bit", "2B-bf16", "9B-8bit"],
        help="要测试的模型列表",
    )
    parser.add_argument(
        "--backends", nargs="+",
        default=["baseline_mlx"],
        choices=["baseline_mlx", "hybrid_ane"],
        help="推理后端",
    )
    parser.add_argument(
        "--prompt-lengths", nargs="+",
        default=["short", "medium", "long"],
        choices=["short", "medium", "long"],
        help="Prompt 长度",
    )
    parser.add_argument(
        "--num-runs", type=int, default=4,
        help="每个配置运行次数（第 1 次为 warmup，取后 N-1 次中位数）",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="追加到已有结果（而非清空重来）",
    )
    parser.add_argument(
        "--no-power", action="store_true",
        help="禁用 PowerMonitor（不采集功耗数据）",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    print(f"Working directory: {ROOT}")
    print(f"Models: {args.models}")
    print(f"Backends: {args.backends}")
    print(f"Prompt lengths: {args.prompt_lengths}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Max generate tokens: {MAX_GENERATE_TOKENS}")
    print()

    if not args.append and JSON_PATH.exists():
        JSON_PATH.unlink()

    results = run_benchmark(args)
    print_summary_table(results)
    print(f"Results saved to:\n  {JSON_PATH}\n  {CSV_PATH}")


if __name__ == "__main__":
    main()
