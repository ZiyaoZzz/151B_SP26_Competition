# CSE 151B Spring 2026 — Math Reasoning Competition

## Hardware & Inference Time

| Item | Details |
|---|---|
| GPU type | NVIDIA A100 40GB (4× for parallel inference) |
| Full private set (943 questions) | ~17 hours on 4× A100 |
| 200-question verification sample | ~3.5 hours on 4× A100 |
| Single-GPU mode (1× A100) | ~68 hours for full set |

The pipeline runs 12-way majority vote with a selective verification pass (free-form questions only), which accounts for most of the runtime.

---

## Model Weights

We fine-tuned `Qwen/Qwen3-4B-Thinking-2507` with GRPO on 227 curated training examples from the public dataset (problems where majority voting at N=12 was wrong).

The fine-tuned model is hosted on HuggingFace Hub:

```
Kevinhhhhhh/qwen3-4b-grpo-v6
```

**Setup — place weights in the `checkpoints/` directory:**

```bash
# Option A: download from HF Hub (requires internet access)
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Kevinhhhhhh/qwen3-4b-grpo-v6",
    local_dir="checkpoints/merged_model_v6",
)
EOF

# Option B: if running on TSCC with HF cache already populated, the model
# is at /tscc/lustre/ddn/scratch/tih009/math_bench/checkpoints/merged_model_v6
# and run_inference.py will accept a local path via --model.
```

After downloading, your directory should contain:
```
checkpoints/
  merged_model_v6/
    config.json
    tokenizer.json
    model-*.safetensors
    ...
```

---

## Reproducing Results — `run_inference()`

### Quick start

```bash
# Install dependencies (vLLM 0.8+ with BitsAndBytes, transformers)
pip install vllm transformers bitsandbytes

# Run full pipeline (4-GPU, uses HF Hub model by default)
python run_inference.py

# Single-GPU mode
python run_inference.py --num-gpus 1

# Custom data or output path
python run_inference.py --data data/private.jsonl --output results/submission.csv

# Use a local checkpoint instead of HF Hub
python run_inference.py --model checkpoints/merged_model_v6
```

### Programmatic call

```python
from run_inference import run_inference

# All defaults — HF Hub model, data/private.jsonl, results/submission.csv, 4 GPUs
run_inference()

# Explicit arguments
run_inference(
    model="Kevinhhhhhh/qwen3-4b-grpo-v6",
    data_path="data/private.jsonl",
    output_path="results/submission.csv",
    num_gpus=4,
)
```

The function returns the path to the written CSV when complete.

---

## Pipeline Details

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Thinking-2507` |
| Fine-tuned checkpoint | `Kevinhhhhhh/qwen3-4b-grpo-v6` (GRPO, ~1 epoch) |
| Majority vote N | 12 |
| Generation temperature | 0.7 |
| Max tokens | 65536 |
| Verification pass | Enabled (free-form questions only; skipped for MCQ) |
| Verification temperature | 0.3 |
| Verification max tokens | 32768 |
| Quantization | BitsAndBytes 4-bit (via vLLM) |

**Selective verification:** We run the model a second time to verify its own free-form answers. MCQ questions are skipped for the verify pass (we found verify hurts MCQ accuracy by ~2.4pp on the validation set).

---

## File Structure

```
151B_SP26_Competition/
├── run_inference.py        # Single entry point — call run_inference() here
├── run_benchmark.py        # Core vLLM inference engine (called by run_inference.py)
├── judger.py               # Answer extraction and scoring logic
├── utils.py                # Utilities used by judger.py
├── cse151b_baseline.ipynb  # Original starter notebook
└── data/
    └── public.jsonl        # Public dataset with ground-truth answers
```

---

## Environment

Tested with:
- Python 3.10
- vLLM 0.8.x
- transformers 4.47+
- bitsandbytes 0.44+
- CUDA 12.x, A100 40GB
