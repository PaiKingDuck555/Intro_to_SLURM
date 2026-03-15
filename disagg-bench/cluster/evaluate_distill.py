#!/usr/bin/env python3
"""
Evaluate teacher vs student (or distilled student) on a held-out dataset split.

Measures:
  - Perplexity (lower = better language modeling)
  - Token-level agreement (% where student top-1 == teacher top-1)

Usage:
  python evaluate_distill.py \
      --teacher facebook/opt-30b \
      --student ./distill_checkpoints/checkpoint_epoch3 \
      --dataset databricks/databricks-dolly-15k \
      --num-samples 500 \
      --output eval_summary.json
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F


def load_model(model_path: str, use_lora_base: str = ""):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if use_lora_base:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            use_lora_base,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(use_lora_base, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    return model, tokenizer


def load_eval_data(dataset_name: str, tokenizer, num_samples: int, seq_length: int):
    from datasets import load_dataset

    if dataset_name == "databricks/databricks-dolly-15k":
        ds = load_dataset(dataset_name, split="train")
        text_key = "context"
        fallback_key = "instruction"
    elif dataset_name == "tatsu-lab/alpaca":
        ds = load_dataset(dataset_name, split="train")
        text_key = "output"
        fallback_key = "instruction"
    else:
        ds = load_dataset(dataset_name, split="train")
        text_key = "text"
        fallback_key = None

    samples = []
    for item in ds:
        text = item.get(text_key, "")
        if not text.strip() and fallback_key:
            text = item.get(fallback_key, "")
        if not text.strip():
            continue

        enc = tokenizer(text, max_length=seq_length, truncation=True, return_tensors="pt")
        if enc["input_ids"].shape[1] >= 16:
            samples.append(enc["input_ids"])
        if len(samples) >= num_samples:
            break

    return samples


@torch.no_grad()
def evaluate_model(model, samples: list, label: str) -> dict:
    total_loss = 0.0
    total_tokens = 0
    all_preds = []

    for i, ids in enumerate(samples):
        ids = ids.to("cuda")
        out = model(ids)
        logits = out.logits.float()

        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += shift_labels.numel()

        preds = shift_logits.argmax(dim=-1)
        all_preds.append(preds.cpu())

        if (i + 1) % 100 == 0:
            print(f"  [{label}] {i+1}/{len(samples)} samples")

    perplexity = math.exp(total_loss / max(total_tokens, 1))
    return {"perplexity": perplexity, "predictions": all_preds, "total_tokens": total_tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--lora-base", default="",
                        help="Base model ID if student is a LoRA adapter")
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--output", default="eval_summary.json")
    args = parser.parse_args()

    print(f"Loading teacher: {args.teacher}")
    teacher_model, tokenizer = load_model(args.teacher)

    print(f"Loading eval data: {args.dataset} ({args.num_samples} samples)")
    samples = load_eval_data(args.dataset, tokenizer, args.num_samples, args.seq_length)
    print(f"  Loaded {len(samples)} valid samples")

    print("Evaluating teacher...")
    t0 = time.perf_counter()
    teacher_result = evaluate_model(teacher_model, samples, "teacher")
    teacher_time = time.perf_counter() - t0

    del teacher_model
    torch.cuda.empty_cache()

    print(f"Loading student: {args.student}")
    student_model, _ = load_model(args.student, use_lora_base=args.lora_base)

    print("Evaluating student...")
    t0 = time.perf_counter()
    student_result = evaluate_model(student_model, samples, "student")
    student_time = time.perf_counter() - t0

    agree = 0
    total = 0
    for t_pred, s_pred in zip(teacher_result["predictions"], student_result["predictions"]):
        agree += (t_pred == s_pred).sum().item()
        total += t_pred.numel()
    token_agreement = agree / max(total, 1)

    summary = {
        "teacher_model": args.teacher,
        "student_model": args.student,
        "dataset": args.dataset,
        "num_samples": len(samples),
        "teacher_perplexity": round(teacher_result["perplexity"], 2),
        "student_perplexity": round(student_result["perplexity"], 2),
        "token_agreement": round(token_agreement, 4),
        "retention_pct": round(token_agreement * 100, 1),
        "teacher_eval_time_s": round(teacher_time, 1),
        "student_eval_time_s": round(student_time, 1),
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"  Teacher perplexity: {summary['teacher_perplexity']}")
    print(f"  Student perplexity: {summary['student_perplexity']}")
    print(f"  Token agreement:    {summary['retention_pct']}%")


if __name__ == "__main__":
    main()
