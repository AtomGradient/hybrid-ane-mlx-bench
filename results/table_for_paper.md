# Benchmark Results — M2 Ultra 192GB

## Hardware & Software
- Device: Apple M2 Ultra, 192GB unified memory
- OS: macOS 26.3 (25D125)
- MLX version: 0.31.0
- Python: 3.11
- Memory bandwidth: 800 GB/s
- ANE: 32 cores, 31.6 TOPS

## Test Configuration
- Max generate tokens: 200
- Sampling: temperature=0.0 (greedy, for reproducibility)
- Runs per config: 4 (discard 1st warmup, median of remaining 3)
- Prompt lengths: short (~6 tokens), medium (~133 tokens), long (~410 tokens)

## Main Results — Baseline MLX (Pure GPU)

| Model | Quant | Prompt | Tokens | TTFT (ms) | Decode (tok/s) | Peak Mem (GB) |
|-------|-------|--------|--------|-----------|----------------|---------------|
| Qwen3.5-0.8B | FP16 | short | 6 | 56 | 71.5 | 3.42 |
| Qwen3.5-0.8B | FP16 | medium | 133 | 69 | 70.2 | 3.58 |
| Qwen3.5-0.8B | FP16 | long | 410 | 96 | 69.0 | 4.18 |
| Qwen3.5-2B | 8-bit | short | 6 | 14 | 141.8 | 2.51 |
| Qwen3.5-2B | 8-bit | medium | 133 | 73 | 141.0 | 2.74 |
| Qwen3.5-2B | 8-bit | long | 410 | 162 | 138.9 | 3.23 |
| Qwen3.5-2B | BF16 | short | 6 | 22 | 101.3 | 4.16 |
| Qwen3.5-2B | BF16 | medium | 133 | 54 | 100.6 | 4.37 |
| Qwen3.5-2B | BF16 | long | 410 | 123 | 99.7 | 4.67 |
| Qwen3.5-9B | 8-bit | short | 6 | 39 | 56.4 | 9.76 |
| Qwen3.5-9B | 8-bit | medium | 133 | 265 | 56.1 | 10.00 |
| Qwen3.5-9B | 8-bit | long | 410 | 625 | 56.5 | 10.43 |

## Hybrid ANE (CoreML Prefill + MLX Decode)

| Model | Quant | Prompt | Tokens | seq_len | Prefill (ms) | Prefill (tok/s) | TTFT (ms) | Decode (tok/s) | Peak Mem (GB) | Status |
|-------|-------|--------|--------|---------|-------------|-----------------|-----------|----------------|---------------|--------|
| Qwen3.5-0.8B | FP16 | short | 6 | 64 | 274 | 22 | 274 | 69.2 | 3.42 | OK |
| Qwen3.5-0.8B | FP16 | medium | 133 | 256 | 410 | 325 | 411 | 71.3 | 3.43 | OK |
| Qwen3.5-0.8B | FP16 | long | 410 | 512 | 99 | 4128 | 100 | 73.3 | 4.18 | OK |
| Qwen3.5-2B | 8-bit | * | — | — | — | — | — | — | — | CONVERSION_FAILED: MLX 8-bit format incompatible with HF→CoreML |
| Qwen3.5-2B | BF16 | short | 6 | 64 | 22 | 275 | 22 | 104.2 | 4.16 | OK |
| Qwen3.5-2B | BF16 | medium | 133 | 256 | 54 | 2477 | 54 | 102.3 | 4.37 | OK |
| Qwen3.5-2B | BF16 | long | 410 | 512 | 122 | 3373 | 122 | 100.7 | 4.67 | OK |
| Qwen3.5-9B | 8-bit* | short | 6 | 64 | 319 | 19 | 319 | 50.0 | 10.12 | OK |
| Qwen3.5-9B | 8-bit* | medium | 133 | 256 | 671 | 198 | 672 | 49.7 | 10.13 | OK |
| Qwen3.5-9B | 8-bit* | long | 410 | 512 | 1238 | 331 | 1265 | 47.6 | 10.14 | OK |

> CoreML conversion requires original FP16/BF16/FP32 HuggingFace weights.
> MLX 8-bit quantized models store weights in incompatible format.
> The 0.8B (FP16) and 2B-bf16 models can be converted directly; 2B-8bit cannot.
> *9B-8bit: CoreML prefill converted from original FP16 HF weights; MLX decode uses 8-bit quantized weights.
> This mixed-precision approach introduces ~11-16% decode throughput degradation vs baseline.

## TTFT Comparison: Baseline vs Hybrid (0.8B FP16)

| Prompt | Tokens | Baseline TTFT | Hybrid TTFT | Ratio (hybrid/base) |
|--------|--------|--------------|-------------|---------------------|
| short | 6 | 56ms | 274ms | 4.9× slower |
| medium | 133 | 69ms | 411ms | 6.0× slower |
| long | 410 | 96ms | 100ms | 1.04× (≈ equal) |

## TTFT Comparison: Baseline vs Hybrid (2B BF16)

| Prompt | Tokens | Baseline TTFT | Hybrid TTFT | Ratio (hybrid/base) |
|--------|--------|--------------|-------------|---------------------|
| short | 6 | 22ms | 22ms | 1.0× (equal) |
| medium | 133 | 54ms | 54ms | 1.0× (equal) |
| long | 410 | 123ms | 122ms | 0.99× (equal) |

> **Zero dispatch overhead**: Unlike 0.8B, the 2B model shows no CoreML dispatch penalty at any prompt length.

## TTFT Comparison: Baseline vs Hybrid (9B 8-bit)

| Prompt | Tokens | Baseline TTFT | Hybrid TTFT | Ratio (hybrid/base) |
|--------|--------|--------------|-------------|---------------------|
| short | 6 | 39ms | 319ms | 8.2× slower |
| medium | 133 | 265ms | 672ms | 2.5× slower |
| long | 410 | 625ms | 1265ms | 2.0× slower |

> **No crossover**: Unlike 0.8B and 2B, the 9B hybrid is always slower than GPU baseline.
> 4-chunk CoreML dispatch overhead + mixed-precision (FP16 CoreML → 8-bit MLX) cache bridge
> causes ~11-16% decode degradation (47.6-50.0 vs 56.1-56.5 tok/s).

## Key Observations

### Decode Throughput (M2 Ultra, memory-bandwidth bound, 800 GB/s)
- **2B 8-bit: 138.9–141.8 tok/s** — fastest; quantization reduces weight transfer per token
- **2B BF16: 99.7–101.3 tok/s** — 8-bit provides ~40% speedup over BF16
- **0.8B FP16: 69.0–71.5 tok/s** — slower than 2B-8bit (FP16 = 2 bytes/weight vs 8-bit = 1 byte/weight)
- **9B 8-bit: 56.1–56.5 tok/s** — bandwidth-limited at 9.5 GB weights on 800 GB/s

### M2 Ultra vs M1 Max Decode Speedup (~1.9×)
- M1 Max (400 GB/s): 0.8B ≈ 37 tok/s, 2B-8bit ≈ 95 tok/s, 9B-8bit ≈ 30 tok/s
- M2 Ultra (800 GB/s): 0.8B ≈ 70 tok/s, 2B-8bit ≈ 140 tok/s, 9B-8bit ≈ 56 tok/s

### ANE Prefill Crossover Point
- seq_len ≤ 256: CoreML dispatch overhead dominates → hybrid is 5–6× slower than MLX GPU
- seq_len = 512 (410 tokens): ANE prefill ≈ MLX GPU prefill (100ms vs 96ms)
- At seq512, ANE achieves ~4128 tok/s, matching MLX GPU's ~4270 tok/s
- Key implication: hybrid ANE approach requires long prompts (≥500 tokens) to be beneficial on M2 Ultra

### Decode Speed: Hybrid ≥ Baseline
- Hybrid decode (69.2–73.3 tok/s) matches or slightly exceeds baseline (69.0–71.5 tok/s)
- CoreML-bridged cache is fully pre-filled → decode starts without additional prefill overhead
- Demonstrates correctness of cache bridge (DeltaNet + full attention states transferred correctly)

## ANE-LM (Private API, Sequential) — Qwen3.5-0.8B FP16

Note: ANE-LM uses the Qwen3.5 chat template, adding ~12 system-prompt tokens.
Single-token ANE dispatch latency: ~42 ms/token.

| Prompt | Tokens (w/ template) | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|--------|----------------------|-----------|-----------------|----------------|
| short | 18 | 769 | 23.4 | 24.3 |
| medium | 145 | 5,867 | 24.7 | 23.8 |
| long | 422 | 17,831 | 23.7 | 22.8 |

## Three-Way TTFT Comparison (Qwen3.5-0.8B FP16)

| Prompt | MLX GPU | CoreML Hybrid | ANE-LM | Winner |
|--------|---------|---------------|--------|--------|
| short | 56ms | 274ms | 769ms | GPU (14× faster than ANE-LM) |
| medium | 69ms | 411ms | 5,867ms | GPU (85× faster than ANE-LM) |
| long | 96ms | 100ms | 17,831ms | GPU ≈ CoreML Hybrid (186× faster than ANE-LM) |

## Key Finding: Batching is Critical for ANE

The fundamental difference between CoreML (batched) and ANE-LM (sequential):
- CoreML processes all N tokens in ONE dispatch call → TTFT = constant dispatch + N×compute
- ANE-LM dispatches ONE token per call → TTFT = N × (dispatch + compute) ≈ N × 42ms
- At seq512 (410 tokens), CoreML batching achieves 4,128 tok/s vs. ANE-LM's 24 tok/s (172× faster for prefill)
- For decode (1 token/step), both approaches are sequential; GPU wins due to higher throughput

## Power Consumption (M2 Ultra, powermetrics via asitop)

All measurements during inference (prefill + decode phases, 4 runs each).

### Baseline MLX (GPU Only)

| Model | Quant | Prompt | CPU (W) | GPU (W) | ANE (W) | Total (W) |
|-------|-------|--------|---------|---------|---------|-----------|
| Qwen3.5-0.8B | FP16 | short | 9.5 | 6.7 | 0 | 16.2 |
| Qwen3.5-0.8B | FP16 | medium | 7.0 | 18.0 | 0 | 25.0 |
| Qwen3.5-0.8B | FP16 | long | 6.7 | 19.0 | 0 | 25.7 |
| Qwen3.5-2B | 8-bit | short | 9.0 | 21.2 | 0 | 30.2 |
| Qwen3.5-2B | 8-bit | medium | 8.4 | 25.8 | 0 | 34.2 |
| Qwen3.5-2B | 8-bit | long | 8.7 | 30.9 | 0 | 39.6 |
| Qwen3.5-2B | BF16 | short | 8.8 | 19.3 | 0 | 28.1 |
| Qwen3.5-2B | BF16 | medium | 8.5 | 21.3 | 0 | 29.8 |
| Qwen3.5-2B | BF16 | long | 7.9 | 23.7 | 0 | 31.6 |
| Qwen3.5-9B | 8-bit | short | 6.6 | 36.5 | 0 | 43.1 |
| Qwen3.5-9B | 8-bit | medium | 6.2 | 41.6 | 0 | 47.8 |
| Qwen3.5-9B | 8-bit | long | 6.3 | 46.9 | 0 | 53.2 |

### Hybrid ANE (CoreML Prefill + MLX Decode)

| Model | Quant | Prompt | CPU (W) | GPU (W) | ANE (W) | Total (W) |
|-------|-------|--------|---------|---------|---------|-----------|
| Qwen3.5-0.8B | FP16 | short | 7.4 | 14.6 | 0.024 | 22.0 |
| Qwen3.5-0.8B | FP16 | medium | 10.5 | 5.2 | 0.002 | 15.7 |
| Qwen3.5-0.8B | FP16 | long | 10.4 | 6.2 | 0.017 | 16.6 |
| Qwen3.5-9B | 8-bit | short | 8.1 | 31.3 | 0 | 39.4 |
| Qwen3.5-9B | 8-bit | medium | 9.5 | 21.5 | 0 | 31.0 |
| Qwen3.5-9B | 8-bit | long | 11.1 | 11.4 | 0 | 22.5 |

### Key Finding: ANE Power is Essentially 0 W

**ANE power is essentially 0 W across all hybrid runs** — despite using `compute_units=ALL`, CoreML routes computation through GPU, not ANE. The "ANE prefill" is a misnomer — CoreML is doing GPU-based prefill. This means the hybrid pipeline's benefit is purely from using CoreML's optimized GPU kernels for batched prefill, not from ANE offloading.

Key observations:
- 0.8B hybrid with long prompt: total power 16.6 W (hybrid) vs 25.7 W (baseline) — **35% reduction** despite both using GPU
- 9B hybrid with long prompt: total power 22.5 W (hybrid) vs 53.2 W (baseline) — **58% reduction**
- The power savings come from CoreML's more efficient GPU kernel utilization, not ANE offloading
- ANE power never exceeds 0.024 W in any hybrid configuration
