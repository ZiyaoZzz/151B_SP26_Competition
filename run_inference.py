#!/usr/bin/env python3
"""
CSE 151B Competition — Single Entry Point

Full pipeline: loads GRPO fine-tuned model, runs inference with N=12 majority vote
and selective verification (free-form questions only), outputs submission CSV.

Usage:
    python run_inference.py                                # use defaults
    python run_inference.py --data data/private.jsonl     # explicit data path
    python run_inference.py --num-gpus 1                  # single-GPU mode

    # Programmatic:
    from run_inference import run_inference
    run_inference()
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

# ── HuggingFace Hub model path ────────────────────────────────────────────────
# After uploading merged_model_v6 to HF Hub:
#   huggingface-cli upload <YOUR_HF_USERNAME>/qwen3-4b-grpo-v6 ./checkpoints/merged_model_v6
# Set this to your uploaded model path:
MODEL_HF_PATH = "Kevinhhhhhh/qwen3-4b-grpo-v6"

# If you want to run with a local checkpoint instead, set this:
MODEL_LOCAL_FALLBACK = "checkpoints/merged_model_v6"

# Final submission hyperparameters (do not change for reproducibility)
NUM_SAMPLES    = 12       # majority vote N
TEMPERATURE    = 0.7      # generation temperature
MAX_TOKENS     = 65536    # max generation tokens
MAX_MODEL_LEN  = 65536    # vLLM context window
VERIFY         = True     # selective verification pass (free-form only)
VERIFY_TEMP    = 0.3      # verification pass temperature
VERIFY_TOKENS  = 32768    # max tokens for verification pass


def run_inference(
    model=None,
    data_path="data/private.jsonl",
    output_path="results/submission.csv",
    num_gpus=4,
    work_dir=None,
):
    """
    Full end-to-end inference pipeline.

    Args:
        model:       HuggingFace Hub model path (defaults to MODEL_HF_PATH above).
                     Falls back to MODEL_LOCAL_FALLBACK if the Hub path is unavailable.
        data_path:   Path to input JSONL dataset (absolute or relative to work_dir).
        output_path: Output CSV path (absolute or relative to work_dir).
        num_gpus:    Number of GPUs to use in parallel (default 4).
                     Use 1 for single-GPU mode (slower but needs only 1 GPU).
        work_dir:    Working directory (default: directory containing this script).

    Returns:
        Path to the written submission CSV.
    """
    if work_dir is None:
        work_dir = Path(__file__).resolve().parent
    work_dir = Path(work_dir)

    if model is None:
        model = MODEL_HF_PATH

    data_path   = Path(data_path)   if os.path.isabs(data_path)   else work_dir / data_path
    output_path = Path(output_path) if os.path.isabs(output_path) else work_dir / output_path

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    benchmark_script = work_dir / "run_benchmark.py"
    if not benchmark_script.exists():
        raise FileNotFoundError(f"run_benchmark.py not found at {benchmark_script}")

    # ── Shard the dataset ─────────────────────────────────────────────────────
    with open(data_path) as f:
        lines = f.readlines()
    total = len(lines)
    print(f"[run_inference] {total} questions, {num_gpus} GPU(s), N={NUM_SAMPLES}, T={TEMPERATURE}, verify={VERIFY}")

    tmp_dir = output_path.parent / "_tmp_shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shard_size = (total + num_gpus - 1) // num_gpus
    shards = []
    for i in range(num_gpus):
        chunk = lines[i * shard_size: (i + 1) * shard_size]
        if not chunk:
            continue
        shard_path = tmp_dir / f"shard_{i}.jsonl"
        with open(shard_path, "w") as f:
            f.writelines(chunk)
        shards.append((i, shard_path))

    # ── Spawn one process per GPU ─────────────────────────────────────────────
    procs = []
    out_paths = []
    log_paths = []
    for gpu_idx, shard_path in shards:
        out_path = tmp_dir / f"out_{gpu_idx}.jsonl"
        log_path = tmp_dir / f"log_{gpu_idx}.txt"
        out_paths.append(out_path)
        log_paths.append(log_path)

        cmd = [
            sys.executable, str(benchmark_script),
            "--gpu",            str(gpu_idx),
            "--model",          model,
            "--data",           str(shard_path),
            "--output",         str(out_path),
            "--sample-frac",    "1.0",
            "--num-samples",    str(NUM_SAMPLES),
            "--temperature",    str(TEMPERATURE),
            "--max-tokens",     str(MAX_TOKENS),
            "--max-model-len",  str(MAX_MODEL_LEN),
            "--no-eval",
        ]
        if VERIFY:
            cmd += [
                "--verify",
                "--verify-temperature", str(VERIFY_TEMP),
                "--verify-max-tokens",  str(VERIFY_TOKENS),
            ]

        log_f = open(log_path, "w")
        p = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, cwd=str(work_dir))
        procs.append((p, log_f))
        print(f"[run_inference] GPU {gpu_idx}: PID {p.pid} | log: {log_path}")

    # ── Wait for all processes ────────────────────────────────────────────────
    failed = []
    for gpu_idx, (p, log_f) in enumerate(procs):
        p.wait()
        log_f.close()
        if p.returncode != 0:
            failed.append(gpu_idx)
            print(f"[run_inference] ERROR: GPU {gpu_idx} exited with code {p.returncode}")
            print(f"  Check log: {log_paths[gpu_idx]}")
    if failed:
        raise RuntimeError(f"Inference failed on GPU(s): {failed}. Check logs above.")

    # ── Merge, sort, and write CSV ────────────────────────────────────────────
    results = []
    for out_path in out_paths:
        if not out_path.exists():
            raise FileNotFoundError(f"Expected output not found: {out_path}")
        with open(out_path) as f:
            for line in f:
                results.append(json.loads(line))
    results.sort(key=lambda r: r["id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for r in results:
            writer.writerow([r["id"], r["response"]])

    print(f"[run_inference] Done. Wrote {len(results)} rows → {output_path}")
    return str(output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="CSE 151B competition inference pipeline")
    p.add_argument("--model",       default=None,
                   help=f"Model path or HF Hub ID (default: {MODEL_HF_PATH})")
    p.add_argument("--data",        default="data/private.jsonl",
                   help="Input JSONL dataset (default: data/private.jsonl)")
    p.add_argument("--output",      default="results/submission.csv",
                   help="Output CSV path (default: results/submission.csv)")
    p.add_argument("--num-gpus",    type=int, default=4,
                   help="Number of GPUs (default: 4)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_inference(
        model=args.model,
        data_path=args.data,
        output_path=args.output,
        num_gpus=args.num_gpus,
    )
