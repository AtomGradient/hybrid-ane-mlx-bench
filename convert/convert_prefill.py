"""Convert Qwen3.5 to CoreML prefill model (.mlpackage).

Loads the HuggingFace model, wraps it in the ANE-compatible model,
traces with torch.jit.trace, then converts to CoreML via coremltools.

Usage::

    source ../../3-11-mlx-community-env/bin/activate
    python -m convert.convert_prefill ../Qwen3.5-0.8B --seq-len 64
    python -m convert.convert_prefill ../Qwen3.5-0.8B --seq-len 128 256 512

Chunked conversion (faster for large seq_len)::

    python -m convert.convert_prefill ../Qwen3.5-0.8B --seq-len 512 --num-chunks 2
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch


def convert_one(
    model_path: str,
    seq_len: int,
    output_dir: str = "models",
    compute_precision: str = "float16",
) -> str:
    """Convert for a single sequence length. Returns the output path."""
    config = json.loads((Path(model_path) / "config.json").read_text())
    model_name = Path(model_path).name

    print(f"\n{'='*60}")
    print(f"Converting {model_name} (seq_len={seq_len})")
    print(f"{'='*60}")

    # 1. Load HF model
    print("1. Loading HF model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    ).eval()
    print(f"   HF model loaded in {time.time()-t0:.1f}s")

    # 2. Build ANE model and load weights
    print("2. Building ANE model...")
    from convert.ane_qwen35 import ANEQwen35Prefill

    ane_model = ANEQwen35Prefill(config, last_logit_only=True, max_seq_len=seq_len)
    ane_model.load_hf_weights(hf_model.state_dict())
    ane_model.eval()
    for p in ane_model.parameters():
        p.requires_grad = False

    # Free HF model memory
    del hf_model
    gc.collect()

    # 3. Trace
    print(f"3. Tracing (seq_len={seq_len})...")
    sample_ids = torch.zeros((1, seq_len), dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(ane_model, (sample_ids,))
    print(f"   Traced in {time.time()-t0:.1f}s")

    # 4. CoreML conversion
    print("4. Converting to CoreML...")
    import coremltools as ct
    from anemll_ext.hybrid_attention_wrapper import HybridPrefillWrapper

    output_names = HybridPrefillWrapper.get_output_names(config)
    ct_outputs = [ct.TensorType(name=name) for name in output_names]

    precision = (
        ct.precision.FLOAT16 if compute_precision == "float16"
        else ct.precision.FLOAT32
    )

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
        ],
        outputs=ct_outputs,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    convert_time = time.time() - t0
    print(f"   CoreML conversion in {convert_time:.1f}s")

    # 5. Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name_slug = model_name.lower().replace(".", "_")
    output_path = str(out_dir / f"ane_{name_slug}_prefill_seq{seq_len}.mlpackage")
    mlmodel.save(output_path)
    print(f"   Saved to {output_path}")

    return output_path


def _convert_chunked(
    model_path: str,
    seq_len: int,
    num_chunks: int,
    output_dir: str = "models",
    compute_precision: str = "float16",
) -> list[str]:
    """Convert the prefill model in chunks. Returns list of output paths."""
    config = json.loads((Path(model_path) / "config.json").read_text())
    model_name = Path(model_path).name
    tc = config.get("text_config", config)
    num_layers = tc["num_hidden_layers"]
    hidden_size = tc["hidden_size"]

    print(f"\n{'='*60}")
    print(f"Converting {model_name} (seq_len={seq_len}, {num_chunks} chunks)")
    print(f"{'='*60}")

    # 1. Load HF weights once
    print("1. Loading HF model weights...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    ).eval()
    hf_state_dict = hf_model.state_dict()
    del hf_model
    gc.collect()
    print(f"   Weights loaded in {time.time()-t0:.1f}s")

    import coremltools as ct
    from anemll_ext.hybrid_attention_wrapper import HybridPrefillWrapper
    from convert.ane_qwen35 import ANEQwen35PrefillChunk

    precision = (
        ct.precision.FLOAT16 if compute_precision == "float16"
        else ct.precision.FLOAT32
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name_slug = model_name.lower().replace(".", "_")

    output_paths = []

    for chunk_idx in range(num_chunks):
        print(f"\n--- Chunk {chunk_idx}/{num_chunks} ---")

        # 2. Build chunk model and load weights
        chunk_model = ANEQwen35PrefillChunk(
            config, chunk_idx, num_chunks,
            last_logit_only=True, max_seq_len=seq_len,
        )
        chunk_model.load_hf_weights(hf_state_dict)
        chunk_model.eval()
        for p in chunk_model.parameters():
            p.requires_grad = False

        # 3. Trace
        is_first = chunk_idx == 0
        if is_first:
            sample_input = torch.zeros((1, seq_len), dtype=torch.long)
            ct_inputs = [
                ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
            ]
        else:
            sample_input = torch.zeros((1, seq_len, hidden_size), dtype=torch.float32)
            ct_inputs = [
                ct.TensorType(name="hidden_states", shape=(1, seq_len, hidden_size)),
            ]

        t0 = time.time()
        with torch.no_grad():
            traced = torch.jit.trace(chunk_model, (sample_input,))
        print(f"   Traced in {time.time()-t0:.1f}s")

        # 4. CoreML conversion
        output_names = HybridPrefillWrapper.get_chunk_output_names(
            config, chunk_idx, num_chunks
        )
        ct_outputs = [ct.TensorType(name=name) for name in output_names]

        t0 = time.time()
        mlmodel = ct.convert(
            traced,
            inputs=ct_inputs,
            outputs=ct_outputs,
            compute_precision=precision,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS17,
            convert_to="mlprogram",
        )
        convert_time = time.time() - t0
        print(f"   CoreML conversion in {convert_time:.1f}s")

        # 5. Save
        output_path = str(
            out_dir / f"ane_{name_slug}_prefill_seq{seq_len}_chunk{chunk_idx}of{num_chunks}.mlpackage"
        )
        mlmodel.save(output_path)
        print(f"   Saved to {output_path}")
        output_paths.append(output_path)

        # Free memory between chunks
        del chunk_model, traced, mlmodel
        gc.collect()

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5 to CoreML prefill model"
    )
    parser.add_argument("model_path", help="HuggingFace model directory")
    parser.add_argument(
        "--seq-len", type=int, nargs="+", default=[64],
        help="Sequence length(s) to convert (default: 64)",
    )
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument(
        "--precision", choices=["float16", "float32"], default="float16",
        help="Compute precision (default: float16)",
    )
    parser.add_argument(
        "--num-chunks", type=int, default=1,
        help="Number of chunks to split the model into (default: 1 = monolithic)",
    )
    args = parser.parse_args()

    for sl in args.seq_len:
        if args.num_chunks > 1:
            _convert_chunked(
                args.model_path,
                sl,
                num_chunks=args.num_chunks,
                output_dir=args.output_dir,
                compute_precision=args.precision,
            )
        else:
            convert_one(
                args.model_path,
                sl,
                output_dir=args.output_dir,
                compute_precision=args.precision,
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
