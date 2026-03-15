#!/usr/bin/env python3
"""
Train a student LLM using offline distillation from cached teacher logits.

The teacher is never in VRAM during training — logits were pre-cached by
cache_teacher_logits.py.

Loss = alpha * CE(student, labels) + (1-alpha) * T^2 * KL(student/T || teacher/T)

The T^2 factor re-scales KL gradients to match CE magnitude (Hinton et al. 2015).

Usage:
  python train_student_distill.py \\
      --student facebook/opt-1.3b \\
      --cache-dir ~/distill_cache \\
      --output-dir ~/distill_checkpoints \\
      --temperature 4.0 --alpha 0.5 --epochs 3
"""

import argparse
import glob
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Loss ──────────────────────────────────────────────────────────────────────

def distillation_loss(
    student_logits: torch.Tensor,   # (B, S, V)
    soft_labels: torch.Tensor,      # (B, S, V) — teacher probs at temperature T
    labels: torch.Tensor,           # (B, S)    — ground truth token ids
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """
    Combined CE + KL distillation loss.
    alpha=1.0 → pure CE. alpha=0.0 → pure KL.
    """
    B, S, V = student_logits.shape

    ce = F.cross_entropy(
        student_logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=-100,
    )

    student_soft = F.log_softmax(student_logits / temperature, dim=-1)  # (B, S, V)
    teacher_soft = soft_labels.to(student_logits.device)

    kl = F.kl_div(
        student_soft.reshape(-1, V),
        teacher_soft.reshape(-1, V),
        reduction="batchmean",
    ) * (temperature ** 2)

    return alpha * ce + (1.0 - alpha) * kl


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_cache_shard(path: str) -> dict:
    # weights_only=False: cache contains non-tensor Python objects (strings, floats)
    # Cache files are written by our own code — not untrusted user data.
    return torch.load(path, map_location="cpu", weights_only=False)


class CachedLogitDataset(Dataset):
    """Loads pre-cached teacher logits from one or more shard files."""

    def __init__(self, shard_paths: list[str]):
        self.samples: list[dict] = []
        for path in shard_paths:
            shard = load_cache_shard(path)
            n = shard["input_ids"].shape[0]
            for i in range(n):
                self.samples.append({
                    "input_ids": shard["input_ids"][i],
                    "soft_labels": shard["soft_labels"][i],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "soft_labels": torch.stack([b["soft_labels"] for b in batch]),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM

    shard_paths = sorted(glob.glob(os.path.join(args.cache_dir, "shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files in {args.cache_dir}")
    print(f"Found {len(shard_paths)} shard(s)")

    print(f"Loading student: {args.student}")
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    dataset = CachedLogitDataset(shard_paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    metrics = []

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to("cuda")   # (B, S)
            soft_labels = batch["soft_labels"]           # (B, S, V)

            # zero_grad BEFORE forward pass (correct order)
            optimizer.zero_grad()

            out = model(input_ids)
            student_logits = out.logits.float()          # (B, S, V)

            # Shift: predict next token
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            loss = distillation_loss(
                student_logits,
                soft_labels.float(),
                labels,
                args.temperature,
                args.alpha,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if step % 50 == 0:
                print(f"  epoch={epoch+1} step={step} loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(len(loader), 1)
        elapsed = time.perf_counter() - t0
        print(f"Epoch {epoch+1}/{args.epochs} — avg_loss={avg_loss:.4f} ({elapsed:.0f}s)")
        metrics.append({"epoch": epoch + 1, "avg_loss": avg_loss, "time_s": round(elapsed, 1)})

        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}")
        model.save_pretrained(ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")

    summary = {
        "student_model": args.student,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "metrics": metrics,
        "final_loss": metrics[-1]["avg_loss"] if metrics else None,
    }
    with open(os.path.join(args.output_dir, "distill_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
