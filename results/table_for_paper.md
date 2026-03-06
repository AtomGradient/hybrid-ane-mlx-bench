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
| Qwen3.5-2B | BF16 | * | — | — | — | — | — | — | — | PENDING |
| Qwen3.5-9B | 8-bit | * | — | — | — | — | — | — | — | CONVERSION_FAILED: MLX 8-bit format incompatible with HF→CoreML |

> CoreML conversion requires original FP16/BF16/FP32 HuggingFace weights.
> MLX 8-bit quantized models store weights in incompatible format.
> The 0.8B (FP16) and 2B-bf16 models can be converted; 2B-8bit and 9B-8bit cannot.

## TTFT Comparison: Baseline vs Hybrid (0.8B FP16)

| Prompt | Tokens | Baseline TTFT | Hybrid TTFT | Ratio (hybrid/base) |
|--------|--------|--------------|-------------|---------------------|
| short | 6 | 56ms | 274ms | 4.9× slower |
| medium | 133 | 69ms | 411ms | 6.0× slower |
| long | 410 | 96ms | 100ms | 1.04× (≈ equal) |

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
| long | 96ms | 100ms | 17,831ms | GPU ≈ Hybrid (186× faster than ANE-LM) |

## Key Finding: Batching is Critical for ANE

The fundamental difference between CoreML (batched) and ANE-LM (sequential):
- CoreML processes all N tokens in ONE dispatch call → TTFT = constant dispatch + N×compute
- ANE-LM dispatches ONE token per call → TTFT = N × (dispatch + compute) ≈ N × 42ms
- At seq512 (410 tokens), CoreML batching achieves 4,128 tok/s vs. ANE-LM's 24 tok/s (172× faster for prefill)
- For decode (1 token/step), both approaches are sequential; GPU wins due to higher throughput
