# Hybrid ANE-MLX-Bench: Disaggregated LLM Inference on Apple Silicon

[![Paper PDF](https://img.shields.io/badge/paper-PDF-red)](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)
[![GitHub Pages](https://img.shields.io/badge/website-live-blue)](https://atomgradient.github.io/hybrid-ane-mlx-bench)

**Paper**: [Disaggregated LLM Inference on Apple Silicon: CoreML ANE Prefill + MLX GPU Decode](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)

**Website**: [atomgradient.github.io/hybrid-ane-mlx-bench](https://atomgradient.github.io/hybrid-ane-mlx-bench)

We benchmark four inference strategies for Qwen3.5 on Apple Silicon, revealing how the Neural Engine (ANE), GPU, and private ANE APIs compare for prefill and decode:

| Strategy | Prefill | Decode | TTFT (long, 410 tok) |
|----------|---------|--------|----------------------|
| **MLX GPU** (baseline) | GPU batched | GPU | **96 ms** |
| **CoreML + MLX** (hybrid) | ANE batched | GPU | **100 ms** |
| **ANE-LM** (private API) | ANE sequential | ANE | 17,831 ms |
| **ANE-LM Hybrid** | ANE sequential | GPU | 17,525 ms |

> **Key finding**: ANE batched prefill (CoreML, seq512) reaches GPU-level throughput at ~410 prompt tokens. Sequential ANE dispatch (private API) caps at ~24 tok/s regardless of prompt length — 3× slower than GPU decode. Replacing ANE decode with GPU decode (ANE-LM Hybrid) gives 3× better decode but cannot fix the sequential-ANE prefill bottleneck.

## Four Approaches

### 1. MLX GPU Baseline
Pure GPU inference via [MLX](https://github.com/ml-explore/mlx). Dynamic shapes, lazy evaluation. Benchmarked across four Qwen3.5 variants (0.8B–9B).

### 2. Hybrid CoreML + MLX
CoreML prefill (ANE, fixed seq_len) → KV-cache bridge → MLX decode (GPU).

- Converts Linear → Conv2d for ANE compatibility
- Custom bridge for Qwen3.5's hybrid DeltaNet + full-attention cache format
- First-load ANE compilation: seq64 ≈ 2 min, seq256 ≈ 50 min, seq512 ≈ 97 min (one-time, cached)

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

### TTFT — Qwen3.5-0.8B (all four pipelines)

| Prompt | Tokens | GPU | Hybrid ANE | ANE-LM | ANE-LM Hybrid |
|--------|--------|-----|-----------|--------|---------------|
| short | 6 / 18* | 56 ms | 274 ms | 769 ms | 744 ms |
| medium | 133 / 145* | 69 ms | 411 ms | 5,867 ms | 5,895 ms |
| long | 410 / 422* | 96 ms | 100 ms | 17,831 ms | 17,525 ms |

*ANE-LM adds chat template tokens (~12 system tokens). ANE-LM Hybrid phases benchmarked independently (prefill: ANE-LM, decode: MLX GPU).

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
- On macOS 26.3, use `compute_units=ALL` (not `CPU_AND_NE` — causes ANE IPC deadlock).
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
@misc{atomgradient2025ane,
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
