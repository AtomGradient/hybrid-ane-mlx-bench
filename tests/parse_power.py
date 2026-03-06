#!/usr/bin/env python3
"""
解析 powermetrics 输出，按时间段对齐到 benchmark 配置，
输出每个配置的平均 ANE/GPU/CPU 功耗。

用法:
    # 在 benchmark 运行期间，另一个终端采集功耗数据
    sudo powermetrics --samplers gpu_power,cpu_power,ane_power -i 500 \
        -o results/powermetrics_raw.txt

    # benchmark 完成后解析
    python tests/parse_power.py results/powermetrics_raw.txt

    # 也可以指定 benchmark 时间戳文件来对齐
    python tests/parse_power.py results/powermetrics_raw.txt \
        --timestamps results/benchmark_timestamps.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PowerSample:
    """单个 powermetrics 采样点。"""
    timestamp_ms: int = 0
    sample_idx: int = 0
    ane_power_mw: float = 0.0
    gpu_power_mw: float = 0.0
    cpu_power_mw: float = 0.0


def parse_powermetrics(filepath: str) -> list[PowerSample]:
    """解析 powermetrics 原始输出文件。

    powermetrics 输出格式示例（macOS 15+）:
        *** Sampled system activity (Thu Mar  5 12:00:00 2026 +0800) (503.12ms elapsed) ***

        **** Processor usage ****
        ...
        CPU Power: 1234 mW
        GPU Power: 5678 mW
        ANE Power: 910 mW
        Combined Power (CPU + GPU + ANE): 7822 mW
    """
    text = Path(filepath).read_text(errors="replace")
    samples: list[PowerSample] = []

    # 按 "*** Sampled system activity" 分割采样块
    blocks = re.split(r"\*{3,}\s*Sampled system activity.*?\*{3,}", text)

    for idx, block in enumerate(blocks):
        if not block.strip():
            continue

        sample = PowerSample(sample_idx=idx)

        # ANE Power
        m = re.search(r"ANE\s+Power:\s*([\d.]+)\s*mW", block, re.IGNORECASE)
        if m:
            sample.ane_power_mw = float(m.group(1))

        # GPU Power
        m = re.search(r"GPU\s+Power:\s*([\d.]+)\s*mW", block, re.IGNORECASE)
        if m:
            sample.gpu_power_mw = float(m.group(1))

        # CPU Power
        m = re.search(r"CPU\s+Power:\s*([\d.]+)\s*mW", block, re.IGNORECASE)
        if m:
            sample.cpu_power_mw = float(m.group(1))

        # 只保留有功耗数据的采样
        if sample.ane_power_mw > 0 or sample.gpu_power_mw > 0 or sample.cpu_power_mw > 0:
            samples.append(sample)

    return samples


def summarize_samples(samples: list[PowerSample]) -> dict:
    """计算采样点的统计摘要。"""
    if not samples:
        return {"count": 0}

    ane = [s.ane_power_mw for s in samples]
    gpu = [s.gpu_power_mw for s in samples]
    cpu = [s.cpu_power_mw for s in samples]
    total = [s.ane_power_mw + s.gpu_power_mw + s.cpu_power_mw for s in samples]

    def stats(vals: list[float]) -> dict:
        avg = sum(vals) / len(vals)
        sorted_v = sorted(vals)
        median = sorted_v[len(sorted_v) // 2]
        return {"avg": round(avg, 1), "median": round(median, 1),
                "min": round(min(vals), 1), "max": round(max(vals), 1)}

    return {
        "count": len(samples),
        "ane_power_mw": stats(ane),
        "gpu_power_mw": stats(gpu),
        "cpu_power_mw": stats(cpu),
        "total_power_mw": stats(total),
    }


def align_to_benchmarks(
    samples: list[PowerSample],
    timestamps_path: str | None = None,
) -> dict[str, dict]:
    """将 powermetrics 数据对齐到 benchmark 配置。

    如果提供了 timestamps 文件（JSON，包含每个配置的 start/end 时间戳），
    则按时间段切分。否则返回整体统计。
    """
    if timestamps_path and Path(timestamps_path).exists():
        ts_data = json.loads(Path(timestamps_path).read_text())
        # timestamps 格式: [{"config": "...", "start_idx": N, "end_idx": M}, ...]
        result = {}
        for entry in ts_data:
            config = entry["config"]
            start = entry.get("start_idx", 0)
            end = entry.get("end_idx", len(samples))
            segment = [s for s in samples if start <= s.sample_idx < end]
            result[config] = summarize_samples(segment)
        return result

    # 无时间戳 → 整体统计
    return {"overall": summarize_samples(samples)}


def print_report(summary: dict[str, dict]):
    """打印功耗报告。"""
    print(f"\n{'='*80}")
    print("POWER METRICS REPORT")
    print(f"{'='*80}")

    for config, stats in summary.items():
        print(f"\n--- {config} ({stats['count']} samples) ---")
        if stats["count"] == 0:
            print("  No data")
            continue

        for metric in ["ane_power_mw", "gpu_power_mw", "cpu_power_mw", "total_power_mw"]:
            s = stats[metric]
            label = metric.replace("_", " ").replace("mw", "(mW)").title()
            print(f"  {label:<25} avg={s['avg']:>8.1f}  "
                  f"median={s['median']:>8.1f}  "
                  f"min={s['min']:>8.1f}  max={s['max']:>8.1f}")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="解析 powermetrics 输出并对齐到 benchmark 配置"
    )
    parser.add_argument("input", help="powermetrics 原始输出文件路径")
    parser.add_argument("--timestamps", default=None,
                        help="benchmark 时间戳 JSON 文件（可选）")
    parser.add_argument("--output", default=None,
                        help="输出 JSON 文件路径（可选）")
    args = parser.parse_args()

    samples = parse_powermetrics(args.input)
    print(f"Parsed {len(samples)} power samples from {args.input}")

    summary = align_to_benchmarks(samples, args.timestamps)
    print_report(summary)

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
