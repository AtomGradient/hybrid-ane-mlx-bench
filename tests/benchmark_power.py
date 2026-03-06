#!/usr/bin/env python3
"""
Power consumption benchmark: ANE prefill vs GPU prefill vs GPU decode.

Uses macOS `powermetrics` (sudo required) to measure GPU / ANE / CPU power (mW).

KEY INSIGHT being validated:
  - During ANE-LM prefill: GPU power ≈ idle
  - During MLX GPU prefill: GPU power ≈ peak
  - During MLX GPU decode:  GPU power ≈ moderate (bandwidth-bound)
  → ANE-LM Hybrid frees the GPU thermal budget during prefill.

Usage:
    sudo python tests/benchmark_power.py --model /path/to/Qwen3.5-0.8B

    # Quick test (just power format check):
    sudo python tests/benchmark_power.py --test

Output:
    results/power_results.json    — per-phase averaged power readings
    /tmp/powermetrics_raw.txt     — full raw powermetrics log
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ANE_LM_BIN = ROOT / "ANE-LM" / "build" / "ane-lm"

PROMPT_LONG = (
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
    "operations to the ANE. Neither framework alone fully exploits the available hardware."
)

# ─── powermetrics log parser ──────────────────────────────────────────────────
# Text format example line headers:
#   *** Sampled system activity (Fri Mar  6 10:23:45 2026 -0800) (100.12ms elapsed) ***
# Power lines (in gpu_power / cpu_power blocks):
#   GPU Power: 1234 mW
#   ANE Power: 456 mW
#   CPU Power: 2987 mW

_RE_HEADER  = re.compile(
    r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\) \*\*\*"
)
_RE_GPU_MW  = re.compile(r"GPU Power:\s+([\d.]+)\s+mW")
_RE_ANE_MW  = re.compile(r"ANE Power:\s+([\d.]+)\s+mW")
_RE_CPU_MW  = re.compile(r"CPU Power:\s+([\d.]+)\s+mW")
_DATETIME_FMT = "%a %b  %d %H:%M:%S %Y"   # "Fri Mar  6 10:23:45 2026"
_DATETIME_FMT2 = "%a %b %d %H:%M:%S %Y"   # "Fri Mar 6 10:23:45 2026" (no padding)


@dataclass
class PowerSample:
    t: float          # unix timestamp
    cpu_mw: float
    gpu_mw: float
    ane_mw: float


def _parse_pm_time(s: str) -> Optional[float]:
    """Parse powermetrics timestamp string to unix time."""
    # Strip timezone suffix for strptime (e.g. " -0800")
    s = re.sub(r"\s+[+-]\d{4}$", "", s.strip())
    for fmt in (_DATETIME_FMT, _DATETIME_FMT2):
        try:
            return datetime.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    return None


def parse_powermetrics_log(log_path: str) -> List[PowerSample]:
    """Parse the full powermetrics text log file into a list of PowerSample."""
    try:
        content = open(log_path, "r", errors="replace").read()
    except OSError:
        return []

    samples: List[PowerSample] = []
    # Split on sample headers; each block is one sample
    parts = _RE_HEADER.split(content)
    # parts = [pre, time_str, elapsed_ms, body, time_str, elapsed_ms, body, ...]
    i = 1
    while i + 2 < len(parts):
        time_str  = parts[i]
        # elapsed_ms = parts[i+1]  # not used
        body      = parts[i + 2]
        t = _parse_pm_time(time_str) or 0.0
        gpu = float(m.group(1)) if (m := _RE_GPU_MW.search(body)) else 0.0
        ane = float(m.group(1)) if (m := _RE_ANE_MW.search(body)) else 0.0
        cpu = float(m.group(1)) if (m := _RE_CPU_MW.search(body)) else 0.0
        samples.append(PowerSample(t=t, cpu_mw=cpu, gpu_mw=gpu, ane_mw=ane))
        i += 3

    return samples


def avg_samples(samples: List[PowerSample], t0: float, t1: float) -> Dict:
    """Average power readings from samples in [t0, t1]."""
    s = [x for x in samples if t0 - 0.05 <= x.t <= t1 + 0.05]
    if not s:
        return {"gpu_mw": 0, "ane_mw": 0, "cpu_mw": 0, "n": 0}
    return {
        "gpu_mw": sum(x.gpu_mw for x in s) / len(s),
        "ane_mw": sum(x.ane_mw for x in s) / len(s),
        "cpu_mw": sum(x.cpu_mw for x in s) / len(s),
        "n": len(s),
    }


# ─── powermetrics process ─────────────────────────────────────────────────────

class PowerMetrics:
    def __init__(self, log_path: str, interval_ms: int = 100):
        self.log_path = log_path
        self.interval_ms = interval_ms
        self._proc: Optional[subprocess.Popen] = None

    def start(self):
        cmd = [
            "sudo", "-n", "powermetrics",
            "--samplers", "cpu_power,gpu_power",
            "-i", str(self.interval_ms),
            "-o", self.log_path,
        ]
        # Open log file fresh
        open(self.log_path, "w").close()
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
        time.sleep(0.8)  # give powermetrics time to write first sample

    def stop(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def read_samples(self) -> List[PowerSample]:
        return parse_powermetrics_log(self.log_path)


def check_sudo() -> bool:
    r = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    return r.returncode == 0


# ─── Inference helpers ────────────────────────────────────────────────────────

def load_mlx(model_path: str):
    from mlx_decode.mlx_model import load_mlx_model, get_language_model
    model, processor = load_mlx_model(model_path)
    lm = get_language_model(model)
    config = json.loads((Path(model_path) / "config.json").read_text())
    return lm, processor.tokenizer, config


def make_cache(config: dict):
    from mlx_lm.models.cache import KVCache, ArraysCache
    tc = config.get("text_config", config)
    n, iv = tc["num_hidden_layers"], tc.get("full_attention_interval", 4)
    return [ArraysCache(size=2) if (i+1) % iv != 0 else KVCache() for i in range(n)]


def mlx_prefill(lm, input_ids, config):
    import mlx.core as mx
    cache = make_cache(config)
    x = mx.array([input_ids])
    out = lm(x, cache=cache)
    logits = out.logits
    mx.eval(logits)
    return int(mx.argmax(logits[0, -1, :]).item()), cache


def mlx_decode(lm, first_token, cache, n, eos_id):
    import mlx.core as mx
    tokens, cur = [], first_token
    for _ in range(n):
        if cur == eos_id:
            break
        tokens.append(cur)
        out = lm(mx.array([[cur]]), cache=cache)
        logits = out.logits
        mx.eval(logits)
        cur = int(mx.argmax(logits[0, -1, :]).item())
    return tokens


def encode_prompt(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text)


# ─── Phase runners (return wall-clock t0, t1) ─────────────────────────────────

def run_mlx_prefill_loop(lm, tokenizer, config, secs: float = 15.0) -> Tuple[float, float, int]:
    """Loop MLX prefill for `secs` seconds. Returns (t0, t1, n_iters)."""
    input_ids = encode_prompt(tokenizer, PROMPT_LONG)
    t0, n = time.time(), 0
    while time.time() - t0 < secs:
        mlx_prefill(lm, input_ids, config)
        n += 1
    t1 = time.time()
    print(f"    {n} prefill iterations in {t1-t0:.1f}s  "
          f"({(t1-t0)/n*1000:.0f}ms each)")
    return t0, t1, n


def run_mlx_decode(lm, tokenizer, config, n: int = 150) -> Tuple[float, float, int]:
    """One prefill (unmeasured) + n decode steps. Returns (t0, t1, n_generated)."""
    tc = config.get("text_config", config)
    eos = tc.get("eos_token_id", 248044)
    input_ids = encode_prompt(tokenizer, PROMPT_LONG)
    first_tok, cache = mlx_prefill(lm, input_ids, config)  # unmeasured
    t0 = time.time()
    tokens = mlx_decode(lm, first_tok, cache, n, eos)
    t1 = time.time()
    print(f"    {len(tokens)} tokens  {len(tokens)/(t1-t0):.1f} tok/s")
    return t0, t1, len(tokens)


def run_anelm_prefill(model_path: str) -> Tuple[float, float, dict]:
    """ANE-LM prefill on long prompt (~17 s). Returns (t0, t1, stats)."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        cache_path = tmp.name
    try:
        t0 = time.time()
        r = subprocess.run(
            [str(ANE_LM_BIN), "prefill",
             "--model", model_path, "--prompt", PROMPT_LONG, "--output", cache_path],
            capture_output=True, text=True, timeout=120,
        )
        t1 = time.time()
        stats = json.loads(r.stdout.strip()) if r.returncode == 0 else {}
        if stats:
            print(f"    {stats['prompt_tokens']} tokens  "
                  f"{stats['prompt_tps']:.1f} tok/s  {stats['prefill_ms']:.0f} ms")
    finally:
        try: os.unlink(cache_path)
        except OSError: pass
    return t0, t1, stats


def run_anelm_hybrid_decode(model_path: str, lm, config, n: int = 150) -> Tuple[float, float, int]:
    """ANE-LM prefill (unmeasured) → cache bridge → MLX decode (measured)."""
    from mlx_decode.anelm_cache_bridge import load_anelm_cache
    tc = config.get("text_config", config)
    eos = tc.get("eos_token_id", 248044)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        cache_path = tmp.name
    try:
        r = subprocess.run(
            [str(ANE_LM_BIN), "prefill",
             "--model", model_path, "--prompt", PROMPT_LONG, "--output", cache_path],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            print(f"    FAILED: {r.stderr[:100]}")
            return 0, 0, 0
        stats = json.loads(r.stdout.strip())
        next_token = stats["next_token"]
        caches, _ = load_anelm_cache(cache_path, config)
    finally:
        try: os.unlink(cache_path)
        except OSError: pass

    t0 = time.time()
    tokens = mlx_decode(lm, next_token, caches, n, eos)
    t1 = time.time()
    print(f"    {len(tokens)} tokens  {len(tokens)/(t1-t0):.1f} tok/s")
    return t0, t1, len(tokens)


def run_anelm_pure(model_path: str, n: int = 50) -> Tuple[float, float]:
    """Full ANE-LM sequential (prefill + decode). Returns (t0, t1)."""
    t0 = time.time()
    subprocess.run(
        [str(ANE_LM_BIN), "generate",
         "--model", model_path, "--prompt", PROMPT_LONG,
         "--max-tokens", str(n), "--temp", "0", "--repeat-penalty", "1.0"],
        capture_output=True, text=True, timeout=120,
    )
    t1 = time.time()
    print(f"    done in {(t1-t0):.1f}s")
    return t0, t1


# ─── Report ───────────────────────────────────────────────────────────────────

def fw(mw: float) -> str:
    return f"{mw/1000:.2f} W"


def print_report(measurements: dict, samples: List[PowerSample]):
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  POWER REPORT — Qwen3.5-0.8B FP16, long prompt (~422 tokens)" + " " * 5 + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  {'Measurement':<34} {'GPU':>8} {'ANE':>8} {'CPU':>8} {'n':>4}  ║")
    print("╟" + "─" * 68 + "╢")

    for key, label in [
        ("mlx_prefill_loop", "MLX GPU prefill (loop, compute-bound)"),
        ("mlx_decode",        "MLX GPU decode  (bandwidth-bound)   "),
        ("anelm_prefill",     "ANE-LM prefill  (private API)       "),
        ("anelm_hybrid_dec",  "ANE-LM Hybrid decode (after bridge) "),
        ("anelm_pure",        "ANE-LM pure     (both phases on ANE)"),
    ]:
        p = measurements.get(key, {})
        if p.get("n", 0) > 0:
            print(f"║  {label:<34} {fw(p['gpu_mw']):>8} "
                  f"{fw(p['ane_mw']):>8} {fw(p['cpu_mw']):>8} {p['n']:>4}  ║")

    print("╠" + "═" * 68 + "╣")

    mlx_pf = measurements.get("mlx_prefill_loop", {})
    ane_pf = measurements.get("anelm_prefill",    {})
    mlx_dc = measurements.get("mlx_decode",       {})
    hyb_dc = measurements.get("anelm_hybrid_dec", {})

    if mlx_pf.get("n", 0) > 0 and ane_pf.get("n", 0) > 0:
        gpu_saved = mlx_pf["gpu_mw"] - ane_pf["gpu_mw"]
        ane_cost  = ane_pf["ane_mw"] - mlx_pf["ane_mw"]
        net       = ane_cost - gpu_saved
        print(f"║  PREFILL: GPU saved by using ANE:    {fw(gpu_saved):>8}" + " " * 22 + "║")
        print(f"║  PREFILL: ANE cost added:            {fw(ane_cost):>8}" + " " * 22 + "║")
        print(f"║  PREFILL: Net package power delta:   {fw(net):>8}  "
              f"({'↑ higher' if net>0 else '↓ lower'} total)" + " " * 9 + "║")
        print("╟" + "─" * 68 + "╢")

    if mlx_dc.get("n", 0) > 0 and hyb_dc.get("n", 0) > 0:
        diff = abs(mlx_dc["gpu_mw"] - hyb_dc["gpu_mw"])
        print(f"║  DECODE:  MLX baseline GPU:          {fw(mlx_dc['gpu_mw']):>8}" + " " * 22 + "║")
        print(f"║  DECODE:  ANE-LM Hybrid GPU:         {fw(hyb_dc['gpu_mw']):>8}" + " " * 22 + "║")
        print(f"║  DECODE:  Difference (expect ~0):    {fw(diff):>8}" + " " * 22 + "║")

    print("╚" + "═" * 68 + "╝")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT.parent / "Qwen3.5-0.8B"))
    parser.add_argument("--interval-ms", type=int, default=100,
                        help="powermetrics sampling interval in ms")
    parser.add_argument("--mlx-prefill-secs", type=float, default=15.0,
                        help="Seconds to loop MLX prefill for sustained power measurement")
    parser.add_argument("--decode-tokens", type=int, default=150)
    parser.add_argument("--log", default="/tmp/powermetrics_raw.txt",
                        help="Path for raw powermetrics output")
    parser.add_argument("--test", action="store_true",
                        help="Quick format test: run powermetrics for 3 s, print parsed samples")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["mlx_prefill", "mlx_decode",
                                 "anelm_prefill", "anelm_decode", "anelm_pure"])
    args = parser.parse_args()

    if not check_sudo():
        print("Error: requires passwordless sudo.")
        print("  Run:  sudo -v")
        print("  Then: sudo python tests/benchmark_power.py --model ...")
        sys.exit(1)

    # ── Quick format test ──
    if args.test:
        print(f"Testing powermetrics format (3 s)... log: {args.log}")
        pm = PowerMetrics(args.log, interval_ms=500)
        pm.start()
        time.sleep(3.5)
        pm.stop()
        samples = pm.read_samples()
        print(f"Parsed {len(samples)} samples:")
        for s in samples[:5]:
            print(f"  t={s.t:.1f}  GPU={s.gpu_mw:.0f}mW  "
                  f"ANE={s.ane_mw:.0f}mW  CPU={s.cpu_mw:.0f}mW")
        if not samples:
            print("\nNo samples parsed — check raw log:")
            os.system(f"head -40 {args.log}")
        return

    # ── Full benchmark ──
    print(f"Power Benchmark — {Path(args.model).name}")
    print(f"Sampling: every {args.interval_ms} ms → {args.log}")
    print()

    print("Loading MLX model...")
    lm, tokenizer, config = load_mlx(args.model)
    print("  Done.")
    print()

    pm = PowerMetrics(args.log, interval_ms=args.interval_ms)
    pm.start()
    print(f"powermetrics running (PID {pm._proc.pid})")
    print()

    phases: Dict[str, Tuple[float, float]] = {}

    def cooldown(s=3.0):
        print(f"  [cooldown {s:.0f}s]\n")
        time.sleep(s)

    # 1. MLX prefill loop
    if "mlx_prefill" not in args.skip:
        print(f"── MLX GPU prefill (loop {args.mlx_prefill_secs:.0f}s) ──")
        t0, t1, _ = run_mlx_prefill_loop(lm, tokenizer, config, args.mlx_prefill_secs)
        phases["mlx_prefill_loop"] = (t0, t1)
        cooldown()

    # 2. MLX decode
    if "mlx_decode" not in args.skip:
        print(f"── MLX GPU decode ({args.decode_tokens} tokens) ──")
        t0, t1, _ = run_mlx_decode(lm, tokenizer, config, args.decode_tokens)
        phases["mlx_decode"] = (t0, t1)
        cooldown()

    # 3. ANE-LM prefill
    if "anelm_prefill" not in args.skip:
        print("── ANE-LM private API prefill ──")
        t0, t1, _ = run_anelm_prefill(args.model)
        phases["anelm_prefill"] = (t0, t1)
        cooldown()

    # 4. ANE-LM Hybrid decode
    if "anelm_decode" not in args.skip:
        print(f"── ANE-LM Hybrid: MLX decode after cache bridge ({args.decode_tokens} tokens) ──")
        t0, t1, _ = run_anelm_hybrid_decode(args.model, lm, config, args.decode_tokens)
        phases["anelm_hybrid_dec"] = (t0, t1)
        cooldown()

    # 5. ANE-LM pure sequential
    if "anelm_pure" not in args.skip:
        print("── ANE-LM pure sequential (prefill + 50 decode tokens) ──")
        t0, t1 = run_anelm_pure(args.model, n=50)
        phases["anelm_pure"] = (t0, t1)

    pm.stop()
    print()
    print("powermetrics stopped. Parsing log...")
    samples = pm.read_samples()
    print(f"  {len(samples)} total samples, time span "
          f"{samples[0].t:.1f}–{samples[-1].t:.1f}" if samples else "  0 samples!")

    # Average power per phase
    measurements = {}
    for key, (t0, t1) in phases.items():
        measurements[key] = avg_samples(samples, t0, t1)

    print_report(measurements, samples)

    # Save
    out_path = ROOT / "results" / "power_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "measurements": measurements,
            "phases": {k: list(v) for k, v in phases.items()},
        }, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Raw log: {args.log}  ({os.path.getsize(args.log)//1024} KB)")


if __name__ == "__main__":
    main()
