#!/usr/bin/env python3
"""
Automated asitop screenshot capture during inference phases.

Launches asitop in a new Terminal window, runs each inference phase,
and captures screenshots of asitop at peak utilization using macOS
screencapture.

Usage:
    # Pre-authenticate sudo, then run:
    sudo -v && python tests/capture_asitop.py --model ../Qwen3.5-0.8B

Screenshots saved to docs/assets/asitop_*.png
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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

OUTPUT_DIR = ROOT / "docs" / "assets"

ANE_LM_BIN = ROOT / "ANE-LM" / "build" / "ane-lm"


# ─── Window management ───────────────────────────────────────────────────────

TERMINAL_APPS = ("iTerm2", "Terminal", "Alacritty", "kitty", "WezTerm")


def launch_asitop(interval: int = 1) -> int:
    """Launch asitop in a new Terminal window, return its CGWindowNumber."""
    # Open a new Terminal window running asitop
    script = f'tell application "Terminal" to do script "sudo asitop --interval {interval}"'
    subprocess.run(["osascript", "-e", script], check=True)
    time.sleep(4)  # wait for asitop to render first frame

    # Bring Terminal to front
    subprocess.run([
        "osascript", "-e",
        'tell application "Terminal" to activate'
    ])
    time.sleep(0.5)

    return _find_asitop_window()


def _find_asitop_window() -> int:
    """Find the Terminal window running asitop.

    Strategy:
      1. Try Quartz CGWindowListCopyWindowInfo (needs pyobjc)
      2. Fall back to CGWindowListCopyWindowInfo via system python
      3. Last resort: full-screen capture (wid=0)
    """
    # Strategy 1: Quartz via current Python
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
        )
        return _quartz_find(CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    except ImportError:
        pass

    # Strategy 2: shell out to system python (has pyobjc built-in)
    try:
        result = subprocess.run(
            ["/usr/bin/python3", "-c", """
import json
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
ws = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
out = []
for w in ws:
    owner = w.get("kCGWindowOwnerName", "") or ""
    name = w.get("kCGWindowName", "") or ""
    if owner == "Terminal":
        out.append({"id": int(w["kCGWindowNumber"]), "name": name})
print(json.dumps(out))
"""],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            windows = json.loads(result.stdout.strip())
            # Prefer window with "asitop" in name
            for w in windows:
                if "asitop" in w["name"].lower():
                    return w["id"]
            # Fallback: first Terminal window
            if windows:
                return windows[0]["id"]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass

    print("  [WARN] Could not find asitop window, using full screen capture")
    return 0


def _quartz_find(CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID) -> int:
    """Find terminal window ID using Quartz API objects."""
    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    )
    candidates = []
    for w in windows:
        owner = w.get("kCGWindowOwnerName", "") or ""
        name = w.get("kCGWindowName", "") or ""
        if owner in TERMINAL_APPS:
            wid = int(w["kCGWindowNumber"])
            if "asitop" in name.lower():
                return wid
            candidates.append(wid)
    if candidates:
        return candidates[0]
    return 0


def capture_window(wid: int, output_path: str):
    """Capture a screenshot of a specific window (or full screen if wid=0)."""
    if wid > 0:
        subprocess.run(
            ["screencapture", "-x", "-o", "-l", str(wid), output_path],
            check=True,
        )
    else:
        subprocess.run(["screencapture", "-x", output_path], check=True)
    size_kb = os.path.getsize(output_path) // 1024
    print(f"  Screenshot saved: {output_path} ({size_kb} KB)")


def close_asitop():
    """Kill asitop process (and its sudo wrapper)."""
    subprocess.run(["pkill", "-f", "asitop"], capture_output=True)


# ─── Inference phases ─────────────────────────────────────────────────────────

def load_mlx(model_path: str):
    from mlx_decode.mlx_model import load_mlx_model, get_language_model
    model, processor = load_mlx_model(model_path)
    lm = get_language_model(model)
    config = json.loads((Path(model_path) / "config.json").read_text())
    return lm, processor.tokenizer, config


def make_cache(config: dict):
    tc = config.get("text_config", config)
    n_layers = tc["num_hidden_layers"]
    interval = tc.get("full_attention_interval", 4)
    from mlx_lm.models.cache import KVCache, ArraysCache
    return [
        ArraysCache(size=2) if (i + 1) % interval != 0 else KVCache()
        for i in range(n_layers)
    ]


def mlx_prefill(lm, input_ids, config):
    import mlx.core as mx
    cache = make_cache(config)
    x = mx.array([input_ids])
    out = lm(x, cache=cache)
    mx.eval(out.logits)
    return int(mx.argmax(out.logits[0, -1, :]).item()), cache


def mlx_decode_loop(lm, first_token, cache, n, eos_id):
    import mlx.core as mx
    cur = first_token
    for _ in range(n):
        if cur == eos_id:
            break
        out = lm(mx.array([[cur]]), cache=cache)
        mx.eval(out.logits)
        cur = int(mx.argmax(out.logits[0, -1, :]).item())


def run_phase_gpu_prefill(lm, tokenizer, config, duration: float = 10.0):
    """Sustain MLX GPU prefill for `duration` seconds."""
    input_ids = tokenizer.encode(PROMPT_LONG)
    t0, n = time.time(), 0
    while time.time() - t0 < duration:
        mlx_prefill(lm, input_ids, config)
        n += 1
    print(f"    {n} prefill iterations in {time.time()-t0:.1f}s")


def run_phase_gpu_decode(lm, tokenizer, config, n_tokens: int = 200):
    """Run MLX GPU decode (after one prefill)."""
    tc = config.get("text_config", config)
    eos = tc.get("eos_token_id", 248044)
    input_ids = tokenizer.encode(PROMPT_LONG)
    first_tok, cache = mlx_prefill(lm, input_ids, config)
    t0 = time.time()
    mlx_decode_loop(lm, first_tok, cache, n_tokens, eos)
    print(f"    {n_tokens} tokens in {time.time()-t0:.1f}s")


def run_phase_ane_prefill(lm, tokenizer, config, coreml_path: str, duration: float = 10.0):
    """Sustain CoreML ANE prefill for `duration` seconds."""
    from engine import HybridInferenceEngine
    engine = HybridInferenceEngine(str(Path(config["_model_path"])), coreml_path)
    input_ids = tokenizer.encode(PROMPT_LONG)
    seq_len = engine._select_seq_len(len(input_ids))
    t0, n = time.time(), 0
    while time.time() - t0 < duration:
        if engine.coreml_chunks:
            engine._prefill_chunked(input_ids, seq_len)
        else:
            engine._prefill_coreml(input_ids, seq_len)
        n += 1
    print(f"    {n} ANE prefill iterations in {time.time()-t0:.1f}s")
    del engine


def run_phase_anelm(model_path: str, duration: float = 15.0):
    """Run ANE-LM sequential (prefill is naturally slow, ~17s for long prompt)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        cache_path = tmp.name
    try:
        subprocess.run(
            [str(ANE_LM_BIN), "prefill",
             "--model", model_path, "--prompt", PROMPT_LONG,
             "--output", cache_path],
            capture_output=True, text=True, timeout=120,
        )
        print(f"    ANE-LM prefill completed")
    finally:
        try:
            os.unlink(cache_path)
        except OSError:
            pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Capture asitop screenshots during inference")
    parser.add_argument("--model", default=str(ROOT.parent / "Qwen3.5-0.8B"),
                        help="Model path for MLX")
    parser.add_argument("--coreml", default=None,
                        help="CoreML .mlpackage path (for ANE prefill phase)")
    parser.add_argument("--interval", type=int, default=1,
                        help="asitop sampling interval (seconds)")
    parser.add_argument("--stabilize", type=float, default=3.0,
                        help="Seconds to wait before capturing screenshot")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Seconds to sustain each phase")
    parser.add_argument("--skip-anelm", action="store_true",
                        help="Skip ANE-LM phase (if binary not available)")
    parser.add_argument("--no-launch", action="store_true",
                        help="Don't launch asitop; find an existing Terminal window running it")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-discover CoreML path
    if args.coreml is None:
        model_name = Path(args.model).name.lower().replace(".", "_").replace("-", "_")
        candidate = ROOT / "models" / f"ane_{model_name.replace('qwen3_5', 'qwen3_5')}_prefill_seq512.mlpackage"
        # Try common patterns
        for pattern in [
            f"ane_qwen3_5-0_8b_prefill_seq512.mlpackage",
            f"ane_qwen3_5-0_8b_prefill_seq512_chunk0of2.mlpackage",
        ]:
            p = ROOT / "models" / pattern
            if p.exists():
                # For chunked: return the base path (engine discovers chunks)
                args.coreml = str(ROOT / "models" / pattern.replace("_chunk0of2", ""))
                break

    print("=" * 60)
    print("Asitop Screenshot Capture")
    print("=" * 60)
    print(f"  Model:     {args.model}")
    print(f"  CoreML:    {args.coreml or 'N/A (skip ANE prefill)'}")
    print(f"  Duration:  {args.duration}s per phase")
    print(f"  Output:    {OUTPUT_DIR}/")
    print()

    # Load model
    print("Loading MLX model...")
    lm, tokenizer, config = load_mlx(args.model)
    config["_model_path"] = args.model
    print("  Done.\n")

    # Find or launch asitop
    if args.no_launch:
        print("Finding existing asitop window...")
        wid = _find_asitop_window()
    else:
        print("Launching asitop...")
        wid = launch_asitop(args.interval)
    print(f"  Window ID: {wid}\n")

    phases = []

    # Phase 0: Idle baseline
    phases.append(("idle", "Idle baseline"))

    # Phase 1: GPU prefill
    phases.append(("gpu_prefill", "MLX GPU prefill (sustained)"))

    # Phase 2: GPU decode
    phases.append(("gpu_decode", "MLX GPU decode"))

    # Phase 3: ANE prefill (if CoreML available)
    if args.coreml:
        phases.append(("ane_prefill", "CoreML ANE prefill (sustained)"))

    # Phase 4: ANE-LM (if binary available)
    if not args.skip_anelm and ANE_LM_BIN.exists():
        phases.append(("anelm_prefill", "ANE-LM private API prefill"))

    for phase_id, phase_name in phases:
        print(f"── Phase: {phase_name} ──")
        out_path = str(OUTPUT_DIR / f"asitop_{phase_id}.png")

        if phase_id == "idle":
            print(f"  Waiting {args.stabilize + 2:.0f}s for idle state...")
            time.sleep(args.stabilize + 2)

        elif phase_id == "gpu_prefill":
            print(f"  Starting GPU prefill loop ({args.duration}s)...")
            # Run in a thread so we can capture mid-phase
            import threading
            t = threading.Thread(
                target=run_phase_gpu_prefill,
                args=(lm, tokenizer, config, args.duration),
            )
            t.start()
            time.sleep(args.stabilize)  # wait for metrics to stabilize

        elif phase_id == "gpu_decode":
            print(f"  Starting GPU decode (200 tokens)...")
            t = threading.Thread(
                target=run_phase_gpu_decode,
                args=(lm, tokenizer, config, 200),
            )
            t.start()
            time.sleep(args.stabilize)

        elif phase_id == "ane_prefill":
            print(f"  Starting ANE prefill loop ({args.duration}s)...")
            t = threading.Thread(
                target=run_phase_ane_prefill,
                args=(lm, tokenizer, config, args.coreml, args.duration),
            )
            t.start()
            time.sleep(args.stabilize)

        elif phase_id == "anelm_prefill":
            print(f"  Starting ANE-LM prefill (long prompt, ~17s)...")
            t = threading.Thread(
                target=run_phase_anelm,
                args=(args.model,),
            )
            t.start()
            time.sleep(args.stabilize)

        # Capture screenshot
        capture_window(wid, out_path)

        # Wait for phase to finish
        if phase_id != "idle":
            t.join()

        # Cooldown
        print(f"  Cooldown 3s...\n")
        time.sleep(3)

    # Cleanup
    if not args.no_launch:
        print("Closing asitop...")
        close_asitop()

    print(f"\nDone! Screenshots saved to {OUTPUT_DIR}/:")
    for f in sorted(OUTPUT_DIR.glob("asitop_*.png")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
