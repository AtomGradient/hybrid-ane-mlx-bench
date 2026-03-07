# Hybrid ANE-MLX-Bench: Disaggregated LLM Inference on Apple Silicon

Please refer here:[ANE batch preill VS MLX](https://github.com/AtomGradient/hybird-batch-prefill-on-ane)

[![Paper PDF](https://img.shields.io/badge/paper-PDF-red)](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)
[![GitHub Pages](https://img.shields.io/badge/website-live-blue)](hybrid-ane-mlx-bench)

**Paper**: [Disaggregated LLM Inference on Apple Silicon: CoreML and ANE Private API Prefill + MLX GPU Decode](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)

**Website**: [atomgradient.github.io/hybrid-ane-mlx-bench](https://atomgradient.github.io/hybrid-ane-mlx-bench)

We benchmark five inference strategies for Qwen3.5 on Apple Silicon, revealing how the Neural Engine (ANE), GPU, and private ANE APIs compare for prefill and decode:

| Strategy | Prefill | Decode | TTFT (long, 410 tok) |
|----------|---------|--------|----------------------|
| **MLX GPU** (baseline) | GPU batched | GPU | **96 ms** |
| **CoreML + MLX** (hybrid) | CoreML batched* | GPU | **100 ms** |
| **ANE-LM** (private API) | ANE sequential | ANE | 17,831 ms |
| **ANE-LM Hybrid** | ANE sequential | GPU | 17,601 ms |
| **ANE-LM Batch** | ANE batch (32 tok) | GPU | ~276 ms |

\* On macOS 26.3, CoreML `compute_units=ALL` routes to GPU, not ANE (ANE power ≈ 0W). See paper Section 4.6.

> **Key findings**: (1) CoreML `compute_units=ALL` on macOS 26.3 routes computation to GPU rather than ANE — the "hybrid ANE" results actually reflect CoreML GPU vs MLX GPU performance. (2) Genuine ANE batch dispatch via private API achieves 268 tok/s (0.8B), an 11.3× speedup over sequential dispatch, proving ANE hardware *can* match GPU-class throughput. (3) ANE prefill reduces GPU power from 62.05W to 0.22W (282× reduction). See companion project: [hybrid-ane-mlx-bench](https://github.com/AtomGradient/hybrid-ane-mlx-bench).

## Four Approaches

### 1. MLX GPU Baseline
Pure GPU inference via [MLX](https://github.com/ml-explore/mlx). Dynamic shapes, lazy evaluation. Benchmarked across four Qwen3.5 variants (0.8B–9B).

### 2. Hybrid CoreML + MLX
CoreML prefill (fixed seq_len) → KV-cache bridge → MLX decode (GPU).

- Converts Linear → Conv2d for ANE compatibility
- Custom bridge for Qwen3.5's hybrid DeltaNet + full-attention cache format
- First-load compilation: seq64 ≈ 2 min, seq256 ≈ 50 min, seq512 ≈ 97 min (one-time, cached)
- **Note**: On macOS 26.3, `compute_units=ALL` routes to GPU, not ANE (measured ANE power ≈ 0W)

### 3. ANE-LM (Private API)
C++/ObjC inference using `AppleNeuralEngine.framework` private APIs directly. See [ANE-LM](https://github.com/AtomGradient/ANE-LM).

Single-token ANE dispatch latency: **~42 ms/token** (independent of model size).

### 4. ANE-LM Hybrid
ANE-LM sequential prefill + MLX GPU decode. ANE-LM and MLX use incompatible KV-cache formats,
so each phase is benchmarked independently: prefill timing from ANE-LM, decode throughput from
the MLX GPU baseline. Results are combined to give end-to-end TTFT and decode measurements.

## Results (M2 Ultra, 192 GB, 800 GB/s)

### Decode throughput — MLX GPU baseline

| Model | Quant | Decode (tok/s) | Peak Mem |
|-------|-------|----------------|----------|
| Qwen3.5-0.8B | FP16 | 69–72 | 3.4–4.2 GB |
| Qwen3.5-2B | 8-bit | 139–142 | 2.5–3.2 GB |
| Qwen3.5-2B | BF16 | 100–101 | 4.2–4.7 GB |
| Qwen3.5-9B | 8-bit | 56–57 | 9.8–10.4 GB |

### TTFT — Qwen3.5-0.8B (all pipelines)

| Prompt | Tokens | GPU | CoreML Hybrid* | ANE-LM | ANE-LM Hybrid | ANE-LM Batch |
|--------|--------|-----|-----------|--------|---------------|-------------|
| short | 6 / 18† | 56 ms | 274 ms | 769 ms | 767 ms | — |
| medium | 133 / 145† | 69 ms | 411 ms | 5,867 ms | 6,060 ms | — |
| long | 410 / 422† | 96 ms | 100 ms | 17,831 ms | 17,601 ms | ~276 ms |

\* CoreML `compute_units=ALL` routes to GPU on macOS 26.3 (ANE power ≈ 0W).
† ANE-LM adds chat template tokens (~12 system tokens). ANE-LM Batch: 268 tok/s prefill at 74 tokens (11.3× speedup over sequential).

### TTFT — Qwen3.5-2B BF16 (Hybrid CoreML + MLX)

| Prompt | Tokens | Baseline GPU | Hybrid ANE | Ratio |
|--------|--------|-------------|-----------|-------|
| short | 6 | 22 ms | 22 ms | 1.0× (equal) |
| medium | 133 | 54 ms | 54 ms | 1.0× (equal) |
| long | 410 | 123 ms | 122 ms | 0.99× (equal) |

The 2B BF16 Hybrid TTFT exactly matches GPU baseline at all prompt lengths — consistent with CoreML routing to GPU on macOS 26.3. Decode: 100–104 tok/s (matches baseline 100–101 tok/s).

### TTFT — Qwen3.5-9B 8-bit (Hybrid CoreML + MLX)

| Prompt | Tokens | Baseline GPU | Hybrid ANE | Ratio |
|--------|--------|-------------|-----------|-------|
| short | 6 | 39 ms | 319 ms | 8.2× slower |
| medium | 133 | 265 ms | 672 ms | 2.5× slower |
| long | 410 | 625 ms | 1,265 ms | 2.0× slower |

The 9B model shows **no crossover point** — CoreML hybrid is always slower than GPU baseline. Contributing factors: 4-chunk CoreML dispatch overhead, and mixed-precision cache bridge (FP16 CoreML prefill → 8-bit MLX decode) causes ~11-16% decode degradation (47.6-50.0 vs 56.1-56.5 tok/s). Since CoreML routes to GPU on macOS 26.3, the overhead is purely from CoreML framework dispatch, not ANE hardware.

## Repository Structure

```
├── engine.py              # HybridInferenceEngine: CoreML prefill + MLX decode
├── sampling.py            # Sampling utilities (temperature, top-p)
├── convert/
│   ├── convert_prefill.py # CoreML model conversion entry point
│   └── ane_qwen35.py      # ANE-compatible Qwen3.5 model (PyTorch)
├── mlx_decode/
│   ├── mlx_model.py       # MLX model loading and decode step
│   └── cache_bridge.py    # CoreML → MLX cache bridge (DeltaNet + KVCache)
├── tests/
│   ├── benchmark.py       # MLX baseline + hybrid ANE benchmark
│   ├── benchmark_ane_lm.py# ANE-LM benchmark wrapper
│   └── parse_power.py     # powermetrics energy data parser
├── results/
│   ├── benchmark_results.json  # All benchmark data (JSON)
│   ├── benchmark_results.csv   # Same data (CSV)
│   └── table_for_paper.md      # Paper-ready tables
└── docs/                  # GitHub Pages site
    ├── index.html         # Interactive results page
    ├── paper.md           # Paper (Markdown)
    └── paper.pdf          # Paper (PDF)
```

## Quickstart

### Requirements

```bash
pip install -r requirements.txt
```

Download a supported model (Qwen3.5-0.8B or Qwen3.5-2B-bf16 in HuggingFace safetensors format):

```bash
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ../Qwen3.5-0.8B
```

### Run MLX baseline inference

```bash
python engine.py ../Qwen3.5-0.8B --prompt "Explain neural networks in one sentence." --max-tokens 100
```

### Run hybrid ANE prefill inference

```bash
# Convert model to CoreML first (one-time, ~20 min for seq64)
python -m convert.convert_prefill ../Qwen3.5-0.8B --seq-len 64 256 512

# Run with CoreML prefill
python engine.py ../Qwen3.5-0.8B \
    --coreml models/ane_qwen3_5-0_8b_prefill_seq512.mlpackage \
    --prompt "Your long prompt here..."
```

### Reproduce benchmarks

```bash
# MLX baseline (all models)
python tests/benchmark.py --backends baseline_mlx \
    --models 0.8B 2B-8bit 2B-bf16 9B-8bit \
    --prompt-lengths short medium long --num-runs 4

# Hybrid ANE (0.8B only, requires CoreML conversion)
python tests/benchmark.py --backends hybrid_ane \
    --models 0.8B --prompt-lengths short medium long --num-runs 4 --append

# ANE-LM (requires building ANE-LM binary separately)
python tests/benchmark_ane_lm.py --runs 4 --append

# ANE-LM Hybrid (measures ANE-LM prefill + MLX GPU decode independently)
python tests/benchmark_ane_lm_hybrid.py --runs 4 --append
```

### CoreML Conversion Notes

- Requires original HF weights (FP16/BF16). MLX 8-bit quantized models are **not** compatible.
- First CoreML load triggers on-device ANE kernel compilation. This is a one-time cost:
  - seq64: ~2 min; seq256: ~50 min; seq512 (2 chunks): ~97 min
- On macOS 26.3, use `compute_units=ALL` (not `CPU_AND_NE` — causes ANE IPC deadlock). Note: `ALL` routes to GPU, not ANE (ANE power ≈ 0W).
- For seq512, use `--num-chunks 2` to split the 28-layer model for faster MIL compilation.

## Hardware

| Spec | M1 Max | M2 Ultra (this work) |
|------|--------|---------------------|
| GPU cores | 32 | 76 |
| ANE cores | 16 | 32 |
| ANE TOPS | 15.8 | 31.6 |
| Memory | 32 GB | 192 GB |
| Bandwidth | 400 GB/s | 800 GB/s |

## Citation

```bibtex
@misc{atomgradient2026ane,
  title  = {Disaggregated LLM Inference on Apple Silicon: CoreML ANE Prefill + MLX GPU Decode},
  author = {AtomGradient},
  year   = {2026},
  url    = {https://github.com/AtomGradient/hybrid-ane-mlx-bench}
}
```

## Acknowledgements

- [maderix/ANE](https://github.com/maderix/ANE) — training on ANE via private APIs
- [ANE-LM](https://github.com/AtomGradient/ANE-LM) — private API inference engine
- [Anemll/Anemll](https://github.com/Anemll/Anemll) — per-layer CoreML conversion for LLMs on ANE
- [MLX](https://github.com/ml-explore/mlx) — Apple's array framework
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — VLM support for MLX
