#!/usr/bin/env python3
"""
CSE 151B Math Reasoning Competition — vLLM Benchmark with Majority Voting

Usage:
    python run_benchmark.py                                  # 1 sample (baseline)
    python run_benchmark.py --num-samples 16                 # 16-way majority vote
    python run_benchmark.py --num-samples 8 --sample-frac 1.0
    python run_benchmark.py --data data/private.jsonl --no-eval --num-samples 16
"""

import argparse
import json
import os
import random
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional

_script_dir = str(Path(__file__).resolve().parent)
for p in (_script_dir, "."):
    if p not in sys.path:
        sys.path.insert(0, p)

_judger_available = False
try:
    from judger import Judger
    _judger_available = True
except Exception:
    print("=" * 60)
    print("WARNING: Failed to import judger. Full traceback:")
    traceback.print_exc()
    print("=" * 60)
    print("Will use built-in fallback scorer.")
    print("=" * 60)

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="vLLM math benchmark with majority voting")
    p.add_argument("--model",   default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--gpu",     default="0")
    p.add_argument("--data",    default="data/public.jsonl")
    p.add_argument("--output",  default="results/starter_results.jsonl")
    p.add_argument("--max-tokens",    type=int, default=65536)
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--gpu-mem",  type=float, default=0.90)
    p.add_argument("--limit",    type=int, default=None)
    p.add_argument("--sample-frac", type=float, default=0.25)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--no-eval",  action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--temperatures", default=None,
                   help="Comma-separated temperatures for multi-temp sampling, e.g. '0.4,0.7,0.9,1.1'. "
                        "Overrides --temperature. N samples split evenly across temperatures.")
    p.add_argument("--top-p",      type=float, default=0.95)
    p.add_argument("--top-k",      type=int,   default=20)
    p.add_argument("--num-samples", type=int, default=1,
                   help="Number of samples per question for majority voting (1 = no voting)")
    p.add_argument("--no-quantize", action="store_true",
                   help="Load model in full precision (skip bitsandbytes 4-bit quantization)")
    p.add_argument("--verify", action="store_true",
                   help="Run a second verification pass using the same model before voting")
    p.add_argument("--verify-max-tokens", type=int, default=32768,
                   help="max_tokens for verification pass (default 32768)")
    p.add_argument("--verify-temperature", type=float, default=0.3,
                   help="Temperature for verification pass (default 0.3)")
    return p.parse_args()

# ── Prompt Construction ──────────────────────────────────────────────────────

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "CRITICAL FORMAT RULES:\n"
    "1. Put ALL final answers in ONE \\boxed{} at the very end — NEVER use separate \\boxed{} for different parts.\n"
    "2. Multiple sub-answers: comma-separated in ONE \\boxed{}: \\boxed{3, 7, -2}.\n"
    "3. If the problem has ONE answer slot but asks for a list of values (all roots, all solutions, etc.), "
    "wrap the entire list in parentheses: \\boxed{(x1, x2, x3)}.\n"
    "4. 'Select all that apply' letter answers: concatenate letters WITHOUT spaces or commas: "
    "\\boxed{BCEG} not \\boxed{B, C, E, G}.\n"
    "5. Express all numerical answers to at least 6 significant figures."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Systematically eliminate each incorrect option before selecting your final answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


SYSTEM_PROMPT_VERIFY_MATH = (
    "You are an expert mathematician checking a proposed solution. "
    "Given a problem and a proposed solution, verify whether it is correct. "
    "If correct, confirm briefly and restate the final answer inside \\boxed{}. "
    "If incorrect, identify the error and provide the correct answer inside \\boxed{}. "
    "CRITICAL FORMAT RULES:\n"
    "1. Put ALL answers in ONE \\boxed{} at the very end — NEVER use separate boxes.\n"
    "2. Multiple sub-answers: comma-separated in ONE \\boxed{}: \\boxed{3, 7, -2}.\n"
    "3. If the problem has ONE answer slot but the answer is a list of values, "
    "wrap in parentheses: \\boxed{(x1, x2, x3)}.\n"
    "4. 'Select all' letters: concatenate WITHOUT spaces or commas: \\boxed{BCEG}.\n"
    "5. Express all numerical answers to at least 6 significant figures."
)

SYSTEM_PROMPT_VERIFY_MCQ = (
    "You are an expert mathematician checking a proposed answer to a multiple-choice problem. "
    "Given the problem, options, and proposed answer, verify whether the answer is correct. "
    "Output ONLY the letter of the correct option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_verify_prompt(question: str, options: Optional[list], response: str) -> list[dict]:
    think_end = response.rfind("</think>")
    answer_section = response[think_end + len("</think>"):].strip() if think_end >= 0 else response.strip()
    # Keep only the tail — boxed answer is always near the end, and truncating avoids
    # exceeding max_model_len when a response was cut off before </think>.
    MAX_CHARS = 6000
    if len(answer_section) > MAX_CHARS:
        answer_section = "[...]\n" + answer_section[-MAX_CHARS:]

    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_content = (
            f"Problem: {question}\n\nOptions:\n{opts_text}\n\n"
            f"Proposed answer: {answer_section}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT_VERIFY_MCQ},
            {"role": "user",   "content": user_content},
        ]

    ans_count = question.count("[ANS]")
    ans_hint = ""
    if ans_count == 1:
        ans_hint = (
            "\n\n[FORMAT: This problem has 1 answer slot. "
            "If the answer is a list of values, use parentheses: \\boxed{(v1, v2, v3)}.]"
        )
    elif ans_count > 1:
        ans_hint = (
            f"\n\n[FORMAT: This problem has {ans_count} answer slots. "
            f"Provide exactly {ans_count} comma-separated answers in ONE \\boxed{{}}.]"
        )

    user_content = f"Problem: {question}{ans_hint}\n\nProposed solution:\n{answer_section}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT_VERIFY_MATH},
        {"role": "user",   "content": user_content},
    ]


def build_prompt(question: str, options: Optional[list]) -> list[dict]:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return [
            {"role": "system", "content": SYSTEM_PROMPT_MCQ},
            {"role": "user",   "content": f"{question}\n\nOptions:\n{opts_text}"},
        ]

    ans_count = question.count("[ANS]")
    user_content = question
    if ans_count == 1:
        user_content += (
            "\n\n[FORMAT NOTE: This problem has 1 answer slot. "
            "If the answer is a list of multiple values (e.g. all roots, all solutions), "
            "wrap them in parentheses: \\boxed{(v1, v2, v3)}. "
            "If it is a single value, use \\boxed{value}.]"
        )
    elif ans_count > 1:
        user_content += (
            f"\n\n[FORMAT NOTE: This problem has {ans_count} answer slots. "
            f"Provide exactly {ans_count} answers, comma-separated in ONE \\boxed{{}}: "
            f"\\boxed{{a, b, ...}}.]"
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT_MATH},
        {"role": "user",   "content": user_content},
    ]

# ── Answer extraction ────────────────────────────────────────────────────────

def extract_boxed_answers(text: str) -> list[str]:
    think_end = text.rfind("</think>")
    text = text[think_end + len("</think>"):] if think_end >= 0 else text

    entries = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth, i = 1, brace_start
        while i < len(text) and depth > 0:
            if text[i] == '{': depth += 1
            elif text[i] == '}': depth -= 1
            i += 1
        if depth == 0:
            content = text[brace_start:i - 1].strip()
            if content:
                entries.append((idx, i, content))
        start = i

    if not entries:
        return []

    group = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = text[entries[j][1]:entries[j + 1][0]]
        if re.match(r'^[\s,\$\.\;\:\-\&\\]*$', gap):
            group.insert(0, entries[j])
        else:
            break
    return [e[2] for e in group]


def extract_answer_key(text: str, is_mcq: bool, num_gold: int = 1) -> str:
    if is_mcq:
        # Strip thinking chain — only look at the final answer section
        think_end = text.rfind("</think>")
        search_text = text[think_end + len("</think>"):] if think_end >= 0 else text
        # Take the LAST boxed letter in the answer section
        matches = re.findall(r"\\boxed\{([A-Za-z])\}", search_text)
        if matches:
            return matches[-1].upper()
        # Fallback: last standalone capital letter in answer section only
        matches = re.findall(r"\b([A-Z])\b", search_text.upper())
        return matches[-1] if matches else ""

    boxed = extract_boxed_answers(text)
    if not boxed:
        return ""

    if len(boxed) > 1:
        return " ||| ".join(boxed)
    elif num_gold > 1:
        return " ||| ".join(_split_top_commas(boxed[0]))
    else:
        return boxed[0]


def _split_top_commas(s: str) -> list[str]:
    depth, parts, cur = 0, [], []
    for ch in s:
        if ch in '({[': depth += 1
        elif ch in ')}]': depth -= 1
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur).strip()); cur = []; continue
        cur.append(ch)
    parts.append(''.join(cur).strip())
    return parts


def _normalize_vote_key(key: str, judger_inst) -> str:
    """Normalize a vote key to reduce splitting on equivalent answers.

    Mirrors the judger's split_by_comma so that \boxed{3, 7} and
    \boxed{3}, \boxed{7} produce the same key instead of splitting votes.
    """
    # Split by ||| first (multiple separate boxes), then by top-level commas
    # within each part (single box with comma-separated sub-answers).
    # This matches judger.auto_judge which always calls split_by_comma.
    atomic = []
    for part in key.split(" ||| "):
        atomic.extend(_split_top_commas(part))

    norm_parts = []
    for part in atomic:
        try:
            norm = judger_inst.norm_math_str(part)
        except Exception:
            norm = part
        # Collapse numeric equivalents (e.g. "0.5" == "1/2" after norm)
        try:
            v = float(norm.replace(" ", ""))
            norm = f"__num_{round(v, 8)}"
        except (ValueError, TypeError):
            pass
        norm_parts.append(norm)
    return " ||| ".join(norm_parts)


def majority_vote(responses: list[str], is_mcq: bool, num_gold: int = 1,
                  judger_inst=None) -> tuple[str, str]:
    keys = []
    for resp in responses:
        key = extract_answer_key(resp, is_mcq, num_gold)
        keys.append(key)

    valid_keys = [k for k in keys if k]
    if not valid_keys:
        return responses[0], ""

    # For free-form with a judger, normalize before counting votes
    if not is_mcq and judger_inst is not None:
        norm_keys = [_normalize_vote_key(k, judger_inst) if k else "" for k in keys]
        norm_valid = [nk for k, nk in zip(keys, norm_keys) if k]
        winner_norm = Counter(norm_valid).most_common(1)[0][0]
        for resp, key, norm_key in zip(responses, keys, norm_keys):
            if norm_key == winner_norm:
                return resp, key
        return responses[0], keys[0] if keys else ""

    winner = Counter(valid_keys).most_common(1)[0][0]
    for resp, key in zip(responses, keys):
        if key == winner:
            return resp, winner
    return responses[0], winner

# ── Scoring ──────────────────────────────────────────────────────────────────

def score_mcq(response: str, gold_letter: str) -> bool:
    key = extract_answer_key(response, is_mcq=True)
    return key == gold_letter.strip().upper()  # gold is always a letter (A-Z)


def _values_match(pred: str, gold: str, tol: float = 1e-6) -> bool:
    for rm in ["\\left", "\\right", "$", "\\,", "\\;"]:
        pred = pred.replace(rm, ""); gold = gold.replace(rm, "")
    pred = pred.strip(); gold = gold.strip()
    if pred == gold:
        return True
    try:
        pv, gv = float(pred), float(gold)
        return abs(pv - gv) <= tol if gv == 0 else abs((pv - gv) / gv) <= tol
    except (ValueError, ZeroDivisionError):
        pass
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import N as sympy_N
        pv, gv = float(sympy_N(parse_latex(pred))), float(sympy_N(parse_latex(gold)))
        return abs(pv - gv) <= tol if gv == 0 else abs((pv - gv) / gv) <= tol
    except Exception:
        pass
    return False


def score_freeform_fallback(response: str, gold_list: list[str]) -> bool:
    boxed = extract_boxed_answers(response)
    if not boxed:
        return False
    preds = boxed if len(boxed) > 1 else (
        _split_top_commas(boxed[0]) if len(gold_list) > 1 else boxed
    )
    if len(preds) != len(gold_list):
        return False
    return all(_values_match(p, g) for p, g in zip(preds, gold_list))

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from tqdm import tqdm

    # ── Load dataset ──────────────────────────────────────────────────────
    data = [json.loads(line) for line in open(args.data)]
    if args.limit:
        data = data[: args.limit]

    full_size = len(data)
    if args.sample_frac < 1.0:
        random.seed(args.seed)
        k = max(1, int(len(data) * args.sample_frac))
        data = random.sample(data, k)
        print(f"Sampled {len(data)} / {full_size} questions ({args.sample_frac:.0%}, seed={args.seed})")

    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    N = args.num_samples

    # Auto-detect if answers exist in the data
    has_answers = "answer" in data[0] if data else False
    if not has_answers and not args.no_eval:
        print("No 'answer' field found in data — forcing --no-eval mode.")
        args.no_eval = True

    print(f"Running {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")
    print(f"Majority voting: {'OFF (1 sample)' if N == 1 else f'{N} samples per question'}")
    print(f"Eval: {'OFF' if args.no_eval else 'ON'}")
    if N > 1:
        print(f"Total generations: {len(data)} × {N} = {len(data) * N}")

    # ── Load model ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=args.model,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
    )
    if not args.no_quantize:
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["load_format"] = "bitsandbytes"
    llm = LLM(**llm_kwargs)

    print("Model loaded.")

    # ── Build prompts ─────────────────────────────────────────────────────
    prompts = []
    for item in data:
        messages = build_prompt(item["question"], item.get("options"))
        prompts.append(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ))

    # ── Generate (single or multi-temperature) ────────────────────────────
    def _make_sp(temp, n):
        return SamplingParams(
            max_tokens=args.max_tokens,
            temperature=temp,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            n=n,
        )

    print(f"Generating responses...")
    if args.temperatures:
        temps = [float(t) for t in args.temperatures.split(",")]
        n_per_temp = [N // len(temps)] * len(temps)
        for i in range(N % len(temps)):
            n_per_temp[i] += 1
        print(f"Multi-temperature: {list(zip(temps, n_per_temp))}")
        responses_list = [[] for _ in range(len(prompts))]
        for temp, n_t in zip(temps, n_per_temp):
            if n_t == 0:
                continue
            batch = llm.generate(prompts, sampling_params=_make_sp(temp, n_t))
            for i, out in enumerate(batch):
                responses_list[i].extend(o.text.strip() for o in out.outputs)
    else:
        raw = llm.generate(prompts, sampling_params=_make_sp(args.temperature, N))
        responses_list = [[o.text.strip() for o in out.outputs] for out in raw]

    # ── Verification pass (optional, free-form only) ─────────────────────
    if args.verify:
        # Only verify free-form questions — verify hurts MCQ (-2.4pp on val)
        verify_prompts_flat = []
        verify_indices = []  # (item_idx, sample_idx) for patching back
        for i, (item, responses) in enumerate(zip(data, responses_list)):
            if item.get("options"):
                continue  # skip MCQ
            for j, resp in enumerate(responses):
                vmsg = build_verify_prompt(item["question"], item.get("options"), resp)
                verify_prompts_flat.append(tokenizer.apply_chat_template(
                    vmsg, tokenize=False, add_generation_prompt=True,
                ))
                verify_indices.append((i, j))

        n_verify = len(verify_prompts_flat)
        print(f"Verification pass: {n_verify} free-form prompts  T={args.verify_temperature}  max_tokens={args.verify_max_tokens}")
        if n_verify > 0:
            verify_sp = SamplingParams(
                max_tokens=args.verify_max_tokens,
                temperature=args.verify_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=0.0,
                n=1,
            )
            verify_raw = llm.generate(verify_prompts_flat, sampling_params=verify_sp)
            verified_flat = [out.outputs[0].text.strip() for out in verify_raw]
            for (i, j), verified_resp in zip(verify_indices, verified_flat):
                responses_list[i][j] = verified_resp
        print("Verification pass complete.")

    # ── Majority vote + score ─────────────────────────────────────────────
    # Create judger for normalization even in no-eval mode (helps vote quality)
    judger_norm = None
    if _judger_available:
        try:
            judger_norm = Judger(strict_extract=False)
            print("Judger loaded for vote normalization.")
        except Exception:
            pass
    judger_eval = judger_norm if not args.no_eval else None

    results = []
    for idx, (item, all_responses) in enumerate(tqdm(
            zip(data, responses_list), total=len(data), desc="Voting & Scoring")):
        is_mcq = bool(item.get("options"))

        # gold may not exist (private set) — use .get()
        gold = item.get("answer")
        num_gold = len(gold) if isinstance(gold, list) else 1

        # Majority vote (with normalization for free-form)
        best_response, winning_key = majority_vote(all_responses, is_mcq, num_gold,
                                                   judger_inst=judger_norm)

        # Score (only when gold exists)
        correct = None
        if not args.no_eval and gold is not None:
            if is_mcq:
                correct = score_mcq(best_response, str(gold))
            else:
                gold_list = gold if isinstance(gold, list) else [gold]
                if judger_eval is not None:
                    try:
                        correct = judger_eval.auto_judge(
                            pred=best_response, gold=gold_list,
                            options=[[]] * len(gold_list),
                        )
                    except Exception:
                        correct = False
                else:
                    correct = score_freeform_fallback(best_response, gold_list)

        result = {
            "id": item.get("id"), "is_mcq": is_mcq,
            "response": best_response,
        }
        if N > 1:
            result["voted_answer"] = winning_key
            result["vote_counts"] = dict(Counter(
                extract_answer_key(r, is_mcq, num_gold) for r in all_responses
            ))
        if not args.no_eval and gold is not None:
            result["gold"] = gold
            result["correct"] = correct

        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    if not args.no_eval:
        scored = [r for r in results if "correct" in r]
        mcq_res  = [r for r in scored if r["is_mcq"]]
        free_res = [r for r in scored if not r["is_mcq"]]
        acc = lambda s: sum(r["correct"] for r in s) / len(s) * 100 if s else 0.0

        print("=" * 50)
        print(f"EVALUATION RESULTS  (N={N} {'majority vote' if N > 1 else 'single sample'})")
        print("=" * 50)
        print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
        print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
        print(f"  Overall    : {sum(r['correct'] for r in scored):4d} / {len(scored):4d}  ({acc(scored):.2f}%)")
        print("=" * 50)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()