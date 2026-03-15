#!/usr/bin/env python3
"""
Cache teacher logits to disk for offline student distillation.

Why offline caching?
  Teacher (30B+) + student share a GPU during online distillation,
  exceeding single-GPU VRAM. We cache teacher soft labels once, then
  the student trains from those tensors — no teacher in VRAM during
  student training.

Usage:
  python cache_teacher_logits.py \\
      --teacher facebook/opt-30b \\
      --student facebook/opt-1.3b \\
      --num-samples 10000 \\
      --seq-length 512 \\
      --temperature 4.0 \\
      --output-dir ~/distill_cache \\
      --shard-id 0 \\
      --num-shards 10 \\
      --quantization int8
"""

import argparse
import os

import torch

from distill_config import DistillConfig


def build_args():
    parser = argparse.ArgumentParser(description="Cache teacher logits for offline distillation")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--quantization",
        choices=["none", "int8", "int4"],
        default="none",
        help="Quantize teacher to reduce VRAM during caching",
    )
    return parser.parse_args()


def load_teacher(model_name: str, quantization: str = "none"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    load_kwargs: dict = {"device_map": "cuda:0", "trust_remote_code": True}
    if quantization == "int8":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["torch_dtype"] = torch.float16
    elif quantization == "int4":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def load_dataset_shard(
    dataset_name: str,
    dataset_config: str,
    tokenizer,
    cfg: DistillConfig,
) -> list:
    """
    Load tokenized samples for this shard.

    NOTE: Short sequences (< 16 tokens) are skipped. The shard counter
    tracks *attempted* samples, not stored ones, so a shard may return
    slightly fewer items than shard_size if the dataset contains many
    short sequences. This is acceptable for distillation datasets.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split="train")
    start, end = cfg.shard_range()

    input_ids_list = []
    attempted = 0

    for item in ds:
        text = item.get("text", "")
        if not text.strip():
            continue
        enc = tokenizer(
            text,
            max_length=cfg.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"]
        if ids.shape[1] >= 16 and attempted >= start:
            input_ids_list.append(ids)
        attempted += 1
        if attempted >= end:
            break

    return input_ids_list


@torch.no_grad()
def compute_soft_labels(
    model, input_ids: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Teacher forward pass → softmax(logits/T) probabilities."""
    ids = input_ids.to("cuda")
    out = model(ids)
    logits = out.logits.float()
    soft = torch.nn.functional.softmax(logits / temperature, dim=-1)
    return soft.cpu()


def main():
    args = build_args()

    cfg = DistillConfig(
        teacher_model=args.teacher,
        student_model=args.student,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        max_seq_length=args.seq_length,
        num_samples=args.num_samples,
        temperature=args.temperature,
        alpha=args.alpha,
        output_dir=args.output_dir,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = cfg.cache_path()

    if os.path.exists(out_path):
        print(f"Cache already exists: {out_path} — skipping.")
        return

    print(f"Loading teacher: {args.teacher} (quant={args.quantization})")
    model, tokenizer = load_teacher(args.teacher, args.quantization)
    vocab_size = model.config.vocab_size

    start, end = cfg.shard_range()
    print(f"Processing shard {args.shard_id}/{args.num_shards}: samples {start}–{end}")

    samples = load_dataset_shard(args.dataset, args.dataset_config, tokenizer, cfg)
    print(f"  Loaded {len(samples)} samples")

    all_input_ids = []
    all_soft_labels = []

    for i, ids in enumerate(samples):
        soft = compute_soft_labels(model, ids, args.temperature)
        all_input_ids.append(ids.cpu())
        all_soft_labels.append(soft)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(samples)}")

    cache = {
        "input_ids": torch.cat(all_input_ids, dim=0),
        "soft_labels": torch.cat(all_soft_labels, dim=0),
        "temperature": args.temperature,
        "model": args.teacher,
        "vocab_size": vocab_size,
    }

    torch.save(cache, out_path)
    size_gb = os.path.getsize(out_path) / (1024 ** 3)
    print(f"Saved {len(samples)} samples → {out_path} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
