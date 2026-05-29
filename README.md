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

We fine-tuned `Qwen/Qwen3-4B-Thinking-2507` with GRPO on 227 curated training examples. The fine-tuned model is on HuggingFace Hub:

```
Kevinhhhhhh/qwen3-4b-grpo-v6
```

**Download weights:**

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Kevinhhhhhh/qwen3-4b-grpo-v6",
    local_dir="checkpoints/merged_model_v6",
)
EOF
```

After downloading:
```
checkpoints/
  merged_model_v6/
    config.json
    tokenizer.json
    model.safetensors
```

---

## Reproducing Results — `run_inference()`

### Quick start

```bash
# Install dependencies
pip install vllm transformers bitsandbytes

# Run full pipeline (4-GPU, uses HF Hub model by default)
python run_inference.py

# Single-GPU mode
python run_inference.py --num-gpus 1

# Explicit paths
python run_inference.py --data data/private.jsonl --output results/submission.csv

# Use a local checkpoint
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

## Inference Pipeline Details

| Parameter | Value | Reason |
|---|---|---|
| Base model | `Qwen/Qwen3-4B-Thinking-2507` | Competition-fixed |
| Fine-tuned checkpoint | `Kevinhhhhhh/qwen3-4b-grpo-v6` | ~1 epoch GRPO |
| Majority vote N | 12 | Best efficiency/accuracy tradeoff |
| Generation temperature | 0.7 | Single-temp beats multi-temp (see below) |
| Max tokens | 65536 | Full context; CoT often hits 20–40K tokens |
| Verification pass | Free-form only (MCQ skipped) | +5.3pp FF, −2.4pp MCQ — selective is net positive |
| Verification temperature | 0.3 | Lower temp = more conservative correction |
| Verification max tokens | 32768 | Answer always near end; 32K is sufficient |
| Quantization | BitsAndBytes 4-bit (via vLLM) | Required to fit on a single 40GB GPU |

### Why selective verification?

After generating N=12 responses, each free-form response is fed back to the model with a verification prompt asking it to check and correct its own answer. MCQ is explicitly skipped.

Measured on 112-question validation set (N=12, T=0.7):

| | No-verify | Full verify | Selective verify |
|---|---|---|---|
| MCQ | 79.5% | 77.1% | **79.5%** |
| Free-form | 55.9% | 61.2% | **61.2%** |
| Overall | 65.2% | ~67.9% | **~68.0%** |

Verification hurts MCQ because the model is already highly confident on letter answers; re-evaluating correct answers risks flipping them. Verification helps free-form because long CoT reasoning sometimes ends with a calculation error in the final boxed step, which a fresh re-read can catch.

### Why not multi-temperature sampling?

We tested generating N=12 samples spread across T=0.4/0.7/0.9/1.1. Result: free-form accuracy dropped −3.0pp vs single T=0.7. The diversity benefit was outweighed by having to reduce `max_tokens` to 32768 to fit the schedule — the model's thinking chains were getting cut short, producing wrong answers even when the reasoning was on track. Single T=0.7 with full 65536-token budget is strictly better.

---

## GRPO Fine-tuning

### What is GRPO?

Group Relative Policy Optimization (GRPO) is an RL algorithm that trains language models to produce correct answers by generating multiple completions per question, scoring each one, and updating the policy to increase the probability of correct completions relative to wrong ones. We used the TRL library's `GRPOTrainer` with several modifications.

### Training Data — How We Selected It

The base model already performs reasonably on easy questions. Training on problems it always gets right provides no signal (all-correct groups give zero gradient). Training on problems it always gets wrong also provides no signal (all-wrong groups give zero gradient under standard GRPO). The useful training signal comes from **intermediate-difficulty problems**: questions where the model sometimes gets it right and sometimes doesn't.

**Curation procedure** (`tools/curate_grpo_data.py`):

1. Run the base model on all 1126 public questions with N=8 completions each (T=0.6).
2. Score each completion individually using `judger.auto_judge()`.
3. Count `n_correct` (how many of the 8 were right).
4. Keep only questions where `1 ≤ n_correct ≤ 7` — at least one correct (not impossible) and at most seven correct (not trivial).

**Result**: 227 questions out of 1126 (20.2%) — the "learnable" subset.

Distribution of `n_correct` in the 227-example training set:

| n_correct | Count | Interpretation |
|---|---|---|
| 1 | 29 | Very hard — model barely solves it |
| 2 | 26 | Hard |
| 3 | 18 | Hard-medium |
| 4 | 22 | Medium (50/50) |
| 5 | 27 | Medium-easy |
| 6 | 35 | Easy-medium |
| 7 | 70 | Easy — model usually gets it right |

Mean n_correct = 4.66 (roughly 58% correct on average across the training set). The training set includes both free-form (123 examples) and MCQ (104 examples).

**Why not DAPO-Math-17k?**

We first tried using DAPO-Math-17k, a published math dataset used in the DAPO paper. It failed entirely: the problems are calibrated for 32B models, so the 4B model was getting nearly everything wrong. All-wrong groups mean every completion scores reward=-1, the group mean is -1, and `reward - mean = 0` for every sample. No gradient flows. The diagnostic is `rewards/reward_fn/mean` staying pinned at -1.0 throughout training.

**Why the competition's own public.jsonl works**: The problems are competition-distribution-aligned by construction, and the 4B model gets ~20% of them in the right difficulty bracket.

### Reward Function

```
+1.0   correct answer (judger confirms match)
-1.0   wrong answer  (judger rejects)  OR  response has no \boxed{} and is short
-0.5   truncated     (response hit the max_completion_length - 50 token limit)
```

The -0.5 truncation penalty is softer than -1.0 because a truncated response may have been on the right track — the model ran out of tokens, not out of reasoning ability. This signal pushes the model toward concise on-budget answers without treating truncation as a hard failure.

We use **sign advantage** instead of standard group-mean centering. Standard GRPO normalizes advantages as `reward - group_mean`. If all 8 rollouts for a prompt are correct, `group_mean = +1`, all advantages = 0, and no gradient flows — "gradient starvation" on easy problems. With sign advantage, we bypass centering entirely and use raw rewards directly as advantages. Every sample carries a non-zero gradient signal regardless of group composition. This is especially important because our base model is already strong on many problems.

### Key Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| Learning rate | 1e-6 | Lower (5e-7) caused near-zero `clip_ratio` — policy barely moved |
| Loss type | `dapo` (token-level) | Sequence-level creates length bias for long CoT; token-level eliminates it |
| β (KL penalty) | 0 | No divergence penalty; let the model explore freely |
| G (rollouts per prompt) | 8 | Memory budget limit on 40GB GPU |
| max_completion_length | 16384 | Qwen3 thinks in long chains; 16K for training, 65K for final inference |
| Temperature (rollout) | 1.0 | Standard for RL training per DeepSeek-R1 and DAPO papers |
| mask_truncated_completions | False | True was silencing 54–69% of gradient per step (see below) |
| scale_rewards | none | Redundant with sign advantage |
| LoRA rank | 16 | Sufficient for refinement of an already-capable model |
| LoRA target modules | all 7 projections | q/k/v/o/gate/up/down |
| Epochs | 2 (wall-killed ~1.06) | 36-hour compute budget |

### Obstacles and Observations During Training

**v5 total failure — diagnosing with `clip_ratio`**

Our first serious training run (v5) produced a model that scored identically to the base model (61.6% → 61.6%). Post-mortem: `clip_ratio/region_mean` stayed at 2e-5 to 1.2e-4 throughout the entire run. This metric measures how often the new policy deviates enough from the reference to trigger PPO clipping — low values mean the policy barely moved from initialization. Root causes:

- LR = 5e-7 was too low (2× lower than GRPO papers recommend for 4B models)
- `loss_type="grpo"` (sequence-level) created length bias: short wrong answers have larger per-token gradients than long reasoning chains, so the model learned to be concise and wrong
- `mask_truncated_completions=True` was silently zeroing the gradient of 54–69% of completions each step, because the model frequently hit the 16K token limit at T=1.0. Nearly zero net gradient per batch.
- Training data (DAPO-Math-17k) was all-wrong for the 4B model → no diversity → no learning

**v6 fix — all four issues corrected simultaneously**

We changed LR to 1e-6, switched to `loss_type="dapo"`, set `mask_truncated_completions=False`, and switched data to `public.jsonl` with the intermediate-difficulty filter. `clip_ratio/region_mean` rose to 5e-5 to 2e-4 — still modest, but 10–100× higher than v5 and consistent with active learning.

**Hard-problem clusters**

During v6 training, every ~40 steps the `rewards/reward_fn/mean` would spike to -1.0 for 1–5 consecutive steps before recovering. These correspond to batches of questions where the model reliably fails (likely complex multi-step integrals or abstract algebra). The spikes were benign — the model would self-correct on subsequent batches. This is expected behavior: the training set includes problems at the hard end (n_correct=1,2) where early in training the model gets everything wrong.

**Memory engineering on 40GB GPUs**

Standard TRL GRPOTrainer goes OOM on 4×A100-40GB with G=8 and 16K completion length. Three sources of memory pressure, each requiring its own fix:

1. Accelerate's `ConvertOutputsToFp32` hook was silently upcasting bf16 logits to fp32, doubling the tensor size from 4.6 GB to 9.3 GB. Permanently no-op'd in `__init__`.
2. vLLM's generation KV cache (~9.5 GB for G=4×16K tokens) was not being freed before the logp forward pass. Added explicit `gc.collect() + torch.cuda.empty_cache()`.
3. Holding all G policy logit tensors in the autograd graph simultaneously costs G×4.6 GB = 18.5 GB — no room for the backward pass. Applied per-completion gradient checkpointing so logits are freed immediately after computing log-probs, then recomputed one at a time during backward.

**Why training on the base model, not the fine-tuned model**

The 227 training examples were curated using the base model's difficulty distribution. If you re-run training from `merged_model_v6` as the starting point, you should re-curate the training data using that model, because its difficulty distribution has shifted — some problems that were intermediate for the base model are now trivial (n_correct=8) or still hard. Otherwise you risk training on problems that are now all-correct or all-wrong for the new starting model.

---

## File Structure

```
151B_SP26_Competition/
├── run_inference.py        # Single entry point — call run_inference() here
├── run_benchmark.py        # Core vLLM inference engine (called by run_inference.py)
├── judger.py               # Answer extraction and scoring logic
├── utils.py                # Utilities used by judger.py
├── submission.csv          # Final v7 submission (943 rows)
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
- trl 1.4.0+ (must be ≥1.4.0 for `loss_type="dapo"`)
- bitsandbytes 0.44+
- CUDA 12.x, A100 40GB
