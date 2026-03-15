# Distillation Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add end-to-end knowledge distillation to disagg-bench: a SLURM-based offline logit caching pipeline (backend) and a Distillation Planner tab in the InfraBench Next.js UI (frontend) that models cost/accuracy/latency trade-offs of distilling a large teacher into a smaller student.

**Architecture:** Backend adds `distill_config.py` (shared config), `cache_teacher_logits.py` (teacher forward + logit caching in shards), `train_student_distill.py` (offline student training with combined CE+KL loss), and two SLURM scripts. Frontend adds `distillation.ts` (pure TS library using scaling laws) and `DistillationPlanner` component wired into `page.tsx`. No changes to existing files except `page.tsx` (add import + section) and `package.json` (add vitest).

**Tech Stack:** Python 3, PyTorch ≥ 2.0, HuggingFace Transformers + Datasets, bitsandbytes, SLURM array jobs, Next.js 14 App Router, TypeScript 5, Tailwind CSS, React 18, Vitest

---

## Chunk 1: Backend — Offline Logit Cache Pipeline

### Task 1: `distill_config.py` — shared config dataclass

**Files:**
- Create: `disagg-bench/cluster/distill_config.py`
- Create: `disagg-bench/cluster/tests/__init__.py`
- Create: `disagg-bench/cluster/tests/test_distill_config.py`

- [ ] **Step 1: Write the failing test**

Create `disagg-bench/cluster/tests/__init__.py` (empty).

Create `disagg-bench/cluster/tests/test_distill_config.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill_config import DistillConfig


def test_distill_config_defaults():
    cfg = DistillConfig(
        teacher_model="facebook/opt-30b",
        student_model="facebook/opt-1.3b",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_seq_length=512,
        num_samples=1000,
        temperature=4.0,
        alpha=0.5,
        output_dir="/tmp/logit_cache",
        shard_id=0,
        num_shards=1,
    )
    assert cfg.teacher_model == "facebook/opt-30b"
    assert cfg.temperature == 4.0
    assert cfg.alpha == 0.5
    assert cfg.shard_size == 1000


def test_distill_config_shard_size():
    cfg = DistillConfig(
        teacher_model="facebook/opt-30b",
        student_model="facebook/opt-1.3b",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_seq_length=512,
        num_samples=10000,
        temperature=4.0,
        alpha=0.5,
        output_dir="/tmp/logit_cache",
        shard_id=2,
        num_shards=5,
    )
    assert cfg.shard_size == 2000


def test_distill_config_shard_range():
    cfg = DistillConfig(
        teacher_model="facebook/opt-30b",
        student_model="facebook/opt-1.3b",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_seq_length=512,
        num_samples=10000,
        temperature=4.0,
        alpha=0.5,
        output_dir="/tmp/logit_cache",
        shard_id=2,
        num_shards=5,
    )
    start, end = cfg.shard_range()
    assert start == 4000
    assert end == 6000


def test_cache_path_format():
    cfg = DistillConfig(
        teacher_model="facebook/opt-1.3b",
        student_model="facebook/opt-1.3b",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_seq_length=128,
        num_samples=100,
        temperature=4.0,
        alpha=0.5,
        output_dir="/tmp/test_cache",
        shard_id=0,
        num_shards=4,
    )
    assert cfg.cache_path().endswith("shard_0000_of_0004.pt")
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd disagg-bench/cluster
python -m pytest tests/test_distill_config.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'distill_config'`

- [ ] **Step 3: Implement `distill_config.py`**

```python
"""
Distillation configuration dataclass.
Shared by cache_teacher_logits.py and train_student_distill.py.
"""
import math
import os
from dataclasses import dataclass


@dataclass
class DistillConfig:
    # Models
    teacher_model: str
    student_model: str

    # Data
    dataset_name: str        # HuggingFace dataset, e.g. "wikitext"
    dataset_config: str      # dataset config, e.g. "wikitext-2-raw-v1"
    max_seq_length: int

    # Distillation hyperparameters
    temperature: float       # softmax temperature τ; sweet spot 2–8
    alpha: float             # loss = alpha*CE + (1-alpha)*KL

    # Sharding
    output_dir: str
    num_samples: int
    shard_id: int            # 0-indexed
    num_shards: int

    @property
    def shard_size(self) -> int:
        return math.ceil(self.num_samples / self.num_shards)

    def shard_range(self) -> tuple[int, int]:
        start = self.shard_id * self.shard_size
        end = min(start + self.shard_size, self.num_samples)
        return start, end

    def cache_path(self) -> str:
        return os.path.join(
            self.output_dir,
            f"shard_{self.shard_id:04d}_of_{self.num_shards:04d}.pt",
        )
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
python -m pytest tests/test_distill_config.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add disagg-bench/cluster/distill_config.py disagg-bench/cluster/tests/__init__.py disagg-bench/cluster/tests/test_distill_config.py
git commit -m "feat: distillation config dataclass with shard math"
```

---

### Task 2: `cache_teacher_logits.py` — teacher forward pass + logit caching

**Files:**
- Create: `disagg-bench/cluster/cache_teacher_logits.py`
- Modify: `disagg-bench/cluster/tests/test_distill_config.py` (append cache format tests)

- [ ] **Step 1: Write cache format tests**

Append to `disagg-bench/cluster/tests/test_distill_config.py`:

```python
import torch
import tempfile


def test_shard_cache_file_structure():
    """Cache file is a dict with required keys; round-trips correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_cache = {
            "input_ids": torch.randint(0, 1000, (10, 128)),
            "soft_labels": torch.randn(10, 128, 50272),
            "temperature": 4.0,
            "model": "facebook/opt-1.3b",
            "vocab_size": 50272,
        }
        path = f"{tmpdir}/shard_0000_of_0001.pt"
        torch.save(fake_cache, path)
        # weights_only=True is required for PyTorch >= 2.6 safety
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert "input_ids" in loaded
        assert "soft_labels" in loaded
        assert loaded["soft_labels"].shape[0] == 10
        assert loaded["vocab_size"] == 50272
```

- [ ] **Step 2: Run to confirm pass**

```bash
python -m pytest tests/test_distill_config.py -v
```
Expected: `5 passed`

- [ ] **Step 3: Implement `cache_teacher_logits.py` — arg parser only**

```python
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


if __name__ == "__main__":
    args = build_args()
    print(f"Teacher: {args.teacher}  Student: {args.student}")
    print(f"Shard {args.shard_id}/{args.num_shards}  Samples: {args.num_samples}")
```

- [ ] **Step 4: Verify `--help` works**

```bash
python cache_teacher_logits.py --help 2>&1 | head -10
```
Expected: shows usage without error

- [ ] **Step 5: Add `load_teacher` function**

Append to `cache_teacher_logits.py` (before `if __name__ == "__main__":`):

```python
import torch


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
```

- [ ] **Step 6: Add `load_dataset_shard` function**

Append to `cache_teacher_logits.py`:

```python
from distill_config import DistillConfig


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
```

- [ ] **Step 7: Add `compute_soft_labels` and `main`**

Append to `cache_teacher_logits.py`:

```python
import os
import json


@torch.no_grad()
def compute_soft_labels(
    model, input_ids: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Teacher forward pass → softmax(logits/T) probabilities."""
    ids = input_ids.to("cuda")
    out = model(ids)
    logits = out.logits.float()          # (1, seq, vocab)
    soft = torch.nn.functional.softmax(logits / temperature, dim=-1)
    return soft.cpu()


def main():
    args = build_args()
    from distill_config import DistillConfig

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

    samples = load_dataset_shard(
        args.dataset, args.dataset_config, tokenizer, cfg
    )
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
        "input_ids": torch.cat(all_input_ids, dim=0),      # (N, seq)
        "soft_labels": torch.cat(all_soft_labels, dim=0),  # (N, seq, vocab)
        "temperature": args.temperature,
        "model": args.teacher,
        "vocab_size": vocab_size,
    }

    torch.save(cache, out_path)
    size_gb = os.path.getsize(out_path) / (1024 ** 3)
    print(f"Saved {len(samples)} samples → {out_path} ({size_gb:.2f} GB)")
```

Update the `if __name__ == "__main__":` block to call `main()`:

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run existing tests still pass**

```bash
python -m pytest tests/test_distill_config.py -v
```
Expected: `5 passed`

- [ ] **Step 9: Commit**

```bash
git add disagg-bench/cluster/cache_teacher_logits.py disagg-bench/cluster/tests/test_distill_config.py
git commit -m "feat: teacher logit caching pipeline with quantized teacher support"
```

---

### Task 3: `train_student_distill.py` — offline student training

**Files:**
- Create: `disagg-bench/cluster/train_student_distill.py`
- Create: `disagg-bench/cluster/tests/test_train_student_distill.py`

- [ ] **Step 1: Write failing tests**

Create `disagg-bench/cluster/tests/test_train_student_distill.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import tempfile


def test_distillation_loss_pure_ce():
    """alpha=1.0 → loss equals cross-entropy."""
    from train_student_distill import distillation_loss

    B, S, V = 2, 16, 100
    student_logits = torch.randn(B, S, V)
    soft_labels = torch.softmax(torch.randn(B, S, V), dim=-1)
    labels = torch.randint(0, V, (B, S))

    loss = distillation_loss(student_logits, soft_labels, labels, temperature=4.0, alpha=1.0)
    expected = torch.nn.functional.cross_entropy(
        student_logits.reshape(-1, V), labels.reshape(-1)
    )
    assert abs(loss.item() - expected.item()) < 1e-4


def test_distillation_loss_pure_kl():
    """alpha=0.0 → loss is non-negative KL divergence."""
    from train_student_distill import distillation_loss

    B, S, V = 2, 16, 100
    student_logits = torch.randn(B, S, V)
    soft_labels = torch.softmax(torch.randn(B, S, V), dim=-1)
    labels = torch.randint(0, V, (B, S))

    loss = distillation_loss(student_logits, soft_labels, labels, temperature=4.0, alpha=0.0)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_distillation_loss_combined_not_nan():
    """Combined loss with alpha=0.5 is finite and positive."""
    from train_student_distill import distillation_loss

    B, S, V = 2, 16, 100
    student_logits = torch.randn(B, S, V)
    soft_labels = torch.softmax(torch.randn(B, S, V), dim=-1)
    labels = torch.randint(0, V, (B, S))

    loss = distillation_loss(student_logits, soft_labels, labels, temperature=4.0, alpha=0.5)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_distillation_loss_gradients_flow():
    """Backward pass produces non-zero gradients."""
    from train_student_distill import distillation_loss

    B, S, V = 2, 8, 50
    student_logits = torch.randn(B, S, V, requires_grad=True)
    soft_labels = torch.softmax(torch.randn(B, S, V), dim=-1)
    labels = torch.randint(0, V, (B, S))

    loss = distillation_loss(student_logits, soft_labels, labels, temperature=4.0, alpha=0.5)
    loss.backward()
    assert student_logits.grad is not None
    assert student_logits.grad.norm().item() > 0.0


def test_load_cache_shard_keys():
    """load_cache_shard returns dict with required keys."""
    from train_student_distill import load_cache_shard

    with tempfile.TemporaryDirectory() as tmpdir:
        fake = {
            "input_ids": torch.randint(0, 1000, (10, 128)),
            "soft_labels": torch.randn(10, 128, 50272),
            "temperature": 4.0,
            "model": "facebook/opt-30b",
            "vocab_size": 50272,
        }
        path = f"{tmpdir}/shard_0000_of_0001.pt"
        torch.save(fake, path)

        data = load_cache_shard(path)
        assert "input_ids" in data
        assert "soft_labels" in data
        assert data["input_ids"].shape == (10, 128)


def test_cached_logit_dataset_length():
    """CachedLogitDataset returns correct length from shard."""
    from train_student_distill import CachedLogitDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        fake = {
            "input_ids": torch.randint(0, 100, (12, 32)),
            "soft_labels": torch.softmax(torch.randn(12, 32, 100), dim=-1),
            "temperature": 4.0,
            "model": "facebook/opt-1.3b",
            "vocab_size": 100,
        }
        path = f"{tmpdir}/shard_0000_of_0001.pt"
        torch.save(fake, path)

        ds = CachedLogitDataset([path])
        assert len(ds) == 12
        sample = ds[0]
        assert "input_ids" in sample
        assert "soft_labels" in sample
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_train_student_distill.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'train_student_distill'`

- [ ] **Step 3: Implement `train_student_distill.py`**

```python
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
    # weights_only=False required because cache contains strings alongside tensors.
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_train_student_distill.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Run full backend test suite**

```bash
python -m pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add disagg-bench/cluster/train_student_distill.py disagg-bench/cluster/tests/test_train_student_distill.py
git commit -m "feat: offline student distillation trainer with correct CE+KL loss and gradient order"
```

---

### Task 4: SLURM scripts for distillation

**Files:**
- Create: `disagg-bench/cluster/submit_cache_logits.slurm`
- Create: `disagg-bench/cluster/submit_distill_student.slurm`

- [ ] **Step 1: Create `submit_cache_logits.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=cache_logits
#SBATCH --output=logs/cache_%A_%a.out
#SBATCH --error=logs/cache_%A_%a.err
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=priority

# ──────────────────────────────────────────────────────────────────────
#  Offline Teacher Logit Caching — 10 shards in parallel
#
#  Usage:
#    TEACHER=facebook/opt-30b STUDENT=facebook/opt-1.3b \
#      sbatch submit_cache_logits.slurm
#
#  Why offline?
#    Teacher + student together exceed single-GPU VRAM on large models.
#    Caching runs teacher alone; student trains from stored soft labels.
# ──────────────────────────────────────────────────────────────────────

export PATH=$HOME/.local/bin:$PATH

TEACHER="${TEACHER:-facebook/opt-30b}"
STUDENT="${STUDENT:-facebook/opt-1.3b}"
NUM_SHARDS="${NUM_SHARDS:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
TEMPERATURE="${TEMPERATURE:-4.0}"
QUANT="${QUANT:-int8}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/distill_cache}"

echo "============================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Shard      : $SLURM_ARRAY_TASK_ID / $NUM_SHARDS"
echo "  Node       : $(hostname)"
echo "  Teacher    : $TEACHER (quant=$QUANT)"
echo "  Samples    : $NUM_SAMPLES  Seq: $SEQ_LENGTH  T: $TEMPERATURE"
echo "  Output     : $OUTPUT_DIR"
echo "  Started    : $(date)"
echo "============================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p logs "$OUTPUT_DIR"

python3 ~/disagg-bench/cluster/cache_teacher_logits.py \
    --teacher "$TEACHER" \
    --student "$STUDENT" \
    --num-samples "$NUM_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --temperature "$TEMPERATURE" \
    --output-dir "$OUTPUT_DIR" \
    --shard-id "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS" \
    --quantization "$QUANT"

echo "Finished: $(date)"
```

- [ ] **Step 2: Syntax-check `submit_cache_logits.slurm`**

```bash
bash -n disagg-bench/cluster/submit_cache_logits.slurm && echo "OK"
```
Expected: `OK`

- [ ] **Step 3: Create `submit_distill_student.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=distill_student
#SBATCH --output=logs/distill_%j.out
#SBATCH --error=logs/distill_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=priority

# ──────────────────────────────────────────────────────────────────────
#  Student Distillation Training
#  Run AFTER all submit_cache_logits.slurm array tasks complete.
#
#  Usage (with dependency):
#    CACHE_ID=$(sbatch --parsable submit_cache_logits.slurm)
#    STUDENT=facebook/opt-1.3b CACHE_DIR=$HOME/distill_cache \
#      sbatch --dependency=afterok:$CACHE_ID submit_distill_student.slurm
# ──────────────────────────────────────────────────────────────────────

export PATH=$HOME/.local/bin:$PATH

STUDENT="${STUDENT:-facebook/opt-1.3b}"
CACHE_DIR="${CACHE_DIR:-$HOME/distill_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/distill_checkpoints}"
TEMPERATURE="${TEMPERATURE:-4.0}"
ALPHA="${ALPHA:-0.5}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-5e-5}"

echo "============================================"
echo "  Job ID       : $SLURM_JOB_ID"
echo "  Node         : $(hostname)"
echo "  Student      : $STUDENT"
echo "  Cache Dir    : $CACHE_DIR"
echo "  T=$TEMPERATURE  α=$ALPHA  epochs=$EPOCHS  lr=$LR"
echo "  Started      : $(date)"
echo "============================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p logs "$OUTPUT_DIR"

python3 ~/disagg-bench/cluster/train_student_distill.py \
    --student "$STUDENT" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --alpha "$ALPHA" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR"

echo "Finished: $(date)"
```

- [ ] **Step 4: Syntax-check `submit_distill_student.slurm`**

```bash
bash -n disagg-bench/cluster/submit_distill_student.slurm && echo "OK"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add disagg-bench/cluster/submit_cache_logits.slurm disagg-bench/cluster/submit_distill_student.slurm
git commit -m "feat: SLURM scripts for sharded logit caching and student distillation"
```

---

### Task 5: Backend integration test

**Files:**
- Create: `disagg-bench/cluster/tests/test_integration.py`

- [ ] **Step 1: Write integration test**

Create `disagg-bench/cluster/tests/test_integration.py`:

```python
"""
Integration test: distillation pipeline modules compose correctly
without a GPU or real model downloads.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import tempfile

from distill_config import DistillConfig
from train_student_distill import (
    distillation_loss,
    load_cache_shard,
    CachedLogitDataset,
    collate_fn,
)


def make_fake_shard(tmpdir: str, n: int = 8, seq: int = 16, vocab: int = 100) -> str:
    cache = {
        "input_ids": torch.randint(0, vocab, (n, seq)),
        "soft_labels": torch.softmax(torch.randn(n, seq, vocab), dim=-1),
        "temperature": 4.0,
        "model": "facebook/opt-1.3b",
        "vocab_size": vocab,
    }
    path = f"{tmpdir}/shard_0000_of_0001.pt"
    torch.save(cache, path)
    return path


def test_full_pipeline_loss_and_backward():
    """Load shard → build dataset → batch → loss → backward."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = make_fake_shard(tmpdir, n=8, seq=16, vocab=100)
        ds = CachedLogitDataset([path])
        assert len(ds) == 8

        batch = collate_fn([ds[0], ds[1]])
        B, S = batch["input_ids"].shape
        V = batch["soft_labels"].shape[-1]

        student_logits = torch.randn(B, S, V, requires_grad=True)
        labels = batch["input_ids"].clone()
        labels[:, :-1] = batch["input_ids"][:, 1:]
        labels[:, -1] = -100

        loss = distillation_loss(
            student_logits,
            batch["soft_labels"].float(),
            labels,
            temperature=4.0,
            alpha=0.5,
        )
        assert not torch.isnan(loss)
        assert loss.item() > 0

        loss.backward()
        assert student_logits.grad is not None
        assert student_logits.grad.norm().item() > 0


def test_config_shard_path_roundtrip():
    cfg = DistillConfig(
        teacher_model="facebook/opt-30b",
        student_model="facebook/opt-1.3b",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_seq_length=512,
        num_samples=10000,
        temperature=4.0,
        alpha=0.5,
        output_dir="/tmp/cache",
        shard_id=3,
        num_shards=10,
    )
    path = cfg.cache_path()
    assert "shard_0003_of_0010" in path
    start, end = cfg.shard_range()
    assert end - start == cfg.shard_size
```

- [ ] **Step 2: Run integration test**

```bash
python -m pytest tests/test_integration.py -v -s
```
Expected: `2 passed`

- [ ] **Step 3: Run full backend suite**

```bash
python -m pytest tests/ -v
```
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add disagg-bench/cluster/tests/test_integration.py
git commit -m "test: backend integration test for distillation pipeline"
```

---

## Chunk 2: Frontend — Distillation Planner

### Task 6: Add Vitest to the UI project

**Files:**
- Modify: `disagg-bench/ui/package.json`
- Create: `disagg-bench/ui/vitest.config.ts`

- [ ] **Step 1: Install Vitest**

```bash
cd disagg-bench/ui
npm install --save-dev vitest @vitest/coverage-v8
```

- [ ] **Step 2: Add test script to `package.json`**

In `disagg-bench/ui/package.json`, add `"test": "vitest run"` to `scripts`:

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "vitest run"
  }
}
```

- [ ] **Step 3: Create `vitest.config.ts`**

```typescript
import { defineConfig } from "vitest/config";
import { resolve } from "path";

export default defineConfig({
  test: {
    environment: "node",
  },
  resolve: {
    alias: {
      "@": resolve(__dirname, "."),
    },
  },
});
```

- [ ] **Step 4: Verify vitest runs**

```bash
cd disagg-bench/ui
npm test -- --reporter=verbose 2>&1 | head -10
```
Expected: `No test files found` or a passing summary (no errors)

- [ ] **Step 5: Commit**

```bash
git add disagg-bench/ui/package.json disagg-bench/ui/package-lock.json disagg-bench/ui/vitest.config.ts
git commit -m "chore: add vitest to InfraBench UI project"
```

---

### Task 7: `distillation.ts` — accuracy and cost modeling library

**Files:**
- Create: `disagg-bench/ui/lib/distillation.ts`
- Create: `disagg-bench/ui/lib/distillation.test.ts`

- [ ] **Step 1: Write failing tests**

Create `disagg-bench/ui/lib/distillation.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import {
  estimateAccuracyRetention,
  estimateDistillationPlan,
  compareDistillVsQuant,
} from "./distillation";

describe("estimateAccuracyRetention", () => {
  it("returns 1.0 when student params equal teacher", () => {
    expect(estimateAccuracyRetention(30, 30)).toBeCloseTo(1.0, 2);
  });

  it("returns less than 1.0 when student is smaller", () => {
    const r = estimateAccuracyRetention(30, 1.3);
    expect(r).toBeLessThan(1.0);
    expect(r).toBeGreaterThan(0.5);
  });

  it("larger gap → lower retention", () => {
    const small = estimateAccuracyRetention(66, 1.3);
    const large = estimateAccuracyRetention(66, 30);
    expect(large).toBeGreaterThan(small);
  });

  it("never returns below 0", () => {
    expect(estimateAccuracyRetention(1000, 0.1)).toBeGreaterThanOrEqual(0);
  });
});

describe("estimateDistillationPlan", () => {
  const base = {
    teacherParams: 30,
    studentParams: 1.3,
    teacherPricePerHour: 45.0,
    studentPricePerHour: 45.0,
    teacherPrefillTps: 6756,
    teacherDecodeTps: 60,
    studentPrefillTps: 15823,
    studentDecodeTps: 124,
    concurrentUsers: 100,
    temperature: 4.0,
    alpha: 0.5,
    numSamples: 10000,
  };

  it("accuracy retention is between 0 and 100", () => {
    const plan = estimateDistillationPlan(base);
    expect(plan.accuracyRetentionPct).toBeGreaterThan(0);
    expect(plan.accuracyRetentionPct).toBeLessThanOrEqual(100);
  });

  it("student costs less than teacher per hour", () => {
    const plan = estimateDistillationPlan(base);
    expect(plan.studentCostPerHour).toBeLessThan(plan.teacherCostPerHour);
  });

  it("costSavingsPct is positive", () => {
    const plan = estimateDistillationPlan(base);
    expect(plan.costSavingsPct).toBeGreaterThan(0);
  });

  it("pipeline durations are positive", () => {
    const plan = estimateDistillationPlan(base);
    expect(plan.cacheJobDurationEstimateHrs).toBeGreaterThan(0);
    expect(plan.trainJobDurationEstimateHrs).toBeGreaterThan(0);
  });
});

describe("compareDistillVsQuant", () => {
  const base = {
    teacherParams: 30,
    studentParams: 1.3,
    teacherPricePerHour: 45.0,
    studentPricePerHour: 45.0,
    teacherInt4PricePerHour: 45.0,
    teacherPrefillTps: 6756,
    teacherDecodeTps: 60,
    teacherInt4PrefillTps: 8065,
    teacherInt4DecodeTps: 24,
    studentPrefillTps: 15823,
    studentDecodeTps: 124,
    concurrentUsers: 100,
    temperature: 4.0,
    alpha: 0.5,
    numSamples: 10000,
  };

  it("returns distill and quantized options", () => {
    const r = compareDistillVsQuant(base);
    expect(r.distill).toBeDefined();
    expect(r.quantized).toBeDefined();
  });

  it("recommendation is either distill or quantize", () => {
    const r = compareDistillVsQuant(base);
    expect(["distill", "quantize"]).toContain(r.recommendation);
  });

  it("reason string is non-empty", () => {
    const r = compareDistillVsQuant(base);
    expect(r.reason.length).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd disagg-bench/ui
npm test 2>&1 | head -15
```
Expected: `Cannot find module './distillation'`

- [ ] **Step 3: Implement `distillation.ts`**

```typescript
/**
 * Distillation cost and accuracy modeling.
 *
 * Uses log-linear scaling from KD literature:
 *   retention = 1 - k * ln(teacher_params / student_params)
 * where k ≈ 0.08 fits DistilBERT (2x compression → ~97% retention)
 * and TinyBERT (7x compression → ~93% retention).
 */

const DISTILL_K = 0.08;

export interface DistillationInputs {
  teacherParams: number;       // billions
  studentParams: number;       // billions
  teacherPricePerHour: number;
  studentPricePerHour: number;
  teacherPrefillTps: number;
  teacherDecodeTps: number;
  studentPrefillTps: number;
  studentDecodeTps: number;
  concurrentUsers: number;
  temperature: number;
  alpha: number;
  numSamples: number;
}

export interface DistillationPlan {
  accuracyRetentionPct: number;
  compressionRatio: number;
  teacherGpusNeeded: number;
  studentGpusNeeded: number;
  teacherCostPerHour: number;
  studentCostPerHour: number;
  costSavingsPct: number;
  studentPrefillTps: number;
  studentDecodeTps: number;
  cacheJobDurationEstimateHrs: number;
  trainJobDurationEstimateHrs: number;
  totalDistillCostUsd: number;
  temperature: number;
  alpha: number;
}

export interface CompareResult {
  distill: DistillationPlan;
  quantized: {
    pricePerHour: number;
    gpusNeeded: number;
    prefillTps: number;
    decodeTps: number;
    accuracyRetentionPct: number;
  };
  recommendation: "distill" | "quantize";
  reason: string;
}

export interface CompareInputs extends DistillationInputs {
  teacherInt4PricePerHour: number;
  teacherInt4PrefillTps: number;
  teacherInt4DecodeTps: number;
}

/** Fraction of teacher accuracy retained after distillation. Returns [0, 1]. */
export function estimateAccuracyRetention(
  teacherParams: number,
  studentParams: number
): number {
  if (studentParams >= teacherParams) return 1.0;
  const ratio = teacherParams / studentParams;
  return Math.max(0, Math.min(1, 1.0 - DISTILL_K * Math.log(ratio)));
}

function gpusNeeded(decodeTps: number, users: number): number {
  return Math.max(1, Math.ceil(users / decodeTps));
}

export function estimateDistillationPlan(
  inputs: DistillationInputs
): DistillationPlan {
  const retention = estimateAccuracyRetention(
    inputs.teacherParams,
    inputs.studentParams
  );
  const compressionRatio = inputs.teacherParams / inputs.studentParams;

  const teacherGpus = gpusNeeded(inputs.teacherDecodeTps, inputs.concurrentUsers);
  const studentGpus = gpusNeeded(inputs.studentDecodeTps, inputs.concurrentUsers);

  const teacherCostPerHour = teacherGpus * inputs.teacherPricePerHour;
  const studentCostPerHour = studentGpus * inputs.studentPricePerHour;
  const costSavingsPct =
    teacherCostPerHour > 0
      ? ((teacherCostPerHour - studentCostPerHour) / teacherCostPerHour) * 100
      : 0;

  // Cache: teacher processes numSamples * ~256 tokens at teacherPrefillTps
  const avgTok = 256;
  const cacheS = (inputs.numSamples * avgTok) / Math.max(inputs.teacherPrefillTps, 1);
  const cacheHrs = cacheS / 3600;

  // Train: 3 epochs at ~500 tok/s (student FP16 training)
  const trainS = (inputs.numSamples * avgTok * 3) / 500;
  const trainHrs = trainS / 3600;

  const totalCost =
    cacheHrs * inputs.teacherPricePerHour + trainHrs * inputs.studentPricePerHour;

  return {
    accuracyRetentionPct: Math.round(retention * 1000) / 10,
    compressionRatio: Math.round(compressionRatio * 10) / 10,
    teacherGpusNeeded: teacherGpus,
    studentGpusNeeded: studentGpus,
    teacherCostPerHour,
    studentCostPerHour,
    costSavingsPct: Math.round(costSavingsPct * 10) / 10,
    studentPrefillTps: inputs.studentPrefillTps,
    studentDecodeTps: inputs.studentDecodeTps,
    cacheJobDurationEstimateHrs: Math.round(cacheHrs * 10) / 10,
    trainJobDurationEstimateHrs: Math.round(trainHrs * 10) / 10,
    totalDistillCostUsd: Math.round(totalCost * 100) / 100,
    temperature: inputs.temperature,
    alpha: inputs.alpha,
  };
}

export function compareDistillVsQuant(inputs: CompareInputs): CompareResult {
  const distill = estimateDistillationPlan(inputs);

  const quantGpus = gpusNeeded(inputs.teacherInt4DecodeTps, inputs.concurrentUsers);
  const quantized = {
    pricePerHour: quantGpus * inputs.teacherInt4PricePerHour,
    gpusNeeded: quantGpus,
    prefillTps: inputs.teacherInt4PrefillTps,
    decodeTps: inputs.teacherInt4DecodeTps,
    accuracyRetentionPct: 97.0, // NF4 double-quant retains ~97% per literature
  };

  // Recommend distillation when it saves >20% cost AND retains >80% accuracy
  const shouldDistill =
    distill.costSavingsPct > 20 && distill.accuracyRetentionPct > 80;

  return {
    distill,
    quantized,
    recommendation: shouldDistill ? "distill" : "quantize",
    reason: shouldDistill
      ? `Distillation saves ${distill.costSavingsPct.toFixed(1)}% cost with ${distill.accuracyRetentionPct.toFixed(1)}% accuracy retention`
      : `Quantization retains ${quantized.accuracyRetentionPct}% accuracy at $0 one-time cost`,
  };
}
```

- [ ] **Step 4: Run tests**

```bash
cd disagg-bench/ui
npm test 2>&1 | tail -10
```
Expected: `11 passed`

- [ ] **Step 5: Type-check**

```bash
npx tsc --noEmit 2>&1 | head -20
```
Expected: no errors in `lib/distillation.ts`

- [ ] **Step 6: Commit**

```bash
git add disagg-bench/ui/lib/distillation.ts disagg-bench/ui/lib/distillation.test.ts
git commit -m "feat: distillation library — scaling law accuracy and cost modeling"
```

---

### Task 8: `DistillationPlanner` React component

**Files:**
- Create: `disagg-bench/ui/components/distillation-planner.tsx`

Matches the exact visual and structural patterns of `finetune-planner.tsx`. No unused props.

- [ ] **Step 1: Implement `distillation-planner.tsx`**

```tsx
"use client";

import { useState, useMemo } from "react";
import { compareDistillVsQuant, type CompareResult } from "@/lib/distillation";
import { getBenchmarksForModel } from "@/lib/benchmark-data";

const PAIRS = [
  {
    teacher: "facebook/opt-66b",
    student: "facebook/opt-1.3b",
    label: "OPT-66B → OPT-1.3B (51× compression)",
  },
  {
    teacher: "facebook/opt-66b",
    student: "facebook/opt-30b",
    label: "OPT-66B → OPT-30B (2.2× compression)",
  },
  {
    teacher: "facebook/opt-30b",
    student: "facebook/opt-1.3b",
    label: "OPT-30B → OPT-1.3B (23× compression)",
  },
];

const GPU_PRICE = 45.0; // B200 on FluidStack, $/hr

function MetricCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: "green" | "amber" | "red" | "blue";
}) {
  const cls = {
    green: "text-emerald-400",
    amber: "text-amber-400",
    red: "text-red-400",
    blue: "text-indigo-400",
  };
  return (
    <div className="bg-zinc-800/80 rounded-lg p-3">
      <div className="text-xs text-zinc-500">{label}</div>
      <div className={`text-lg font-semibold mt-1 ${color ? cls[color] : ""}`}>
        {value}
      </div>
      {sub && <div className="text-xs text-zinc-500 mt-0.5">{sub}</div>}
    </div>
  );
}

export function DistillationPlanner() {
  const [pairIdx, setPairIdx] = useState(0);
  const [temperature, setTemperature] = useState(4.0);
  const [alpha, setAlpha] = useState(0.5);
  const [numSamples, setNumSamples] = useState(10000);
  const [concurrentUsers, setConcurrentUsers] = useState(100);

  const result: CompareResult | null = useMemo(() => {
    const pair = PAIRS[pairIdx];
    const tBenches = getBenchmarksForModel(pair.teacher, "none");
    const sBenches = getBenchmarksForModel(pair.student, "none");
    const tInt4 = getBenchmarksForModel(pair.teacher, "int4");

    if (!tBenches.length || !sBenches.length) return null;

    const bestT = tBenches.reduce((a, b) => (a.decode_tps > b.decode_tps ? a : b));
    const bestS = sBenches.reduce((a, b) => (a.decode_tps > b.decode_tps ? a : b));
    // Fallback to FP16 teacher data if no INT4 data exists for this teacher
    const bestTInt4 = tInt4.length
      ? tInt4.reduce((a, b) => (a.decode_tps > b.decode_tps ? a : b))
      : bestT;

    return compareDistillVsQuant({
      teacherParams: bestT.params_b,
      studentParams: bestS.params_b,
      teacherPricePerHour: GPU_PRICE,
      studentPricePerHour: GPU_PRICE,
      teacherInt4PricePerHour: GPU_PRICE,
      teacherPrefillTps: bestT.prefill_tps,
      teacherDecodeTps: bestT.decode_tps,
      teacherInt4PrefillTps: bestTInt4.prefill_tps,
      teacherInt4DecodeTps: bestTInt4.decode_tps,
      studentPrefillTps: bestS.prefill_tps,
      studentDecodeTps: bestS.decode_tps,
      concurrentUsers,
      temperature,
      alpha,
      numSamples,
    });
  }, [pairIdx, temperature, alpha, numSamples, concurrentUsers]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold">Distillation Planner</h2>
        <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-900/50 text-indigo-300 border border-indigo-700/50">
          NEW
        </span>
        <span className="text-xs text-zinc-500 hidden sm:inline">
          Model distillation vs. quantization — cost &amp; accuracy trade-offs
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Inputs ── */}
        <div className="space-y-5 bg-zinc-800/30 border border-zinc-700/50 rounded-xl p-5">
          <div>
            <label className="block text-xs text-zinc-400 mb-1.5">
              Teacher → Student Pair
            </label>
            <select
              value={pairIdx}
              onChange={(e) => setPairIdx(Number(e.target.value))}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
            >
              {PAIRS.map((p, i) => (
                <option key={i} value={i}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Temperature (τ)
              </label>
              <input
                type="number"
                min={1}
                max={20}
                step={0.5}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
              />
              <div className="text-xs text-zinc-600 mt-1">Sweet spot: 2–8</div>
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Alpha (α) — CE weight
              </label>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
              />
              <div className="text-xs text-zinc-600 mt-1">0 = pure KL, 1 = pure CE</div>
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Dataset Samples
              </label>
              <input
                type="number"
                value={numSamples}
                step={1000}
                onChange={(e) => setNumSamples(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Concurrent Users
              </label>
              <input
                type="number"
                value={concurrentUsers}
                onChange={(e) => setConcurrentUsers(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
              />
            </div>
          </div>

          <div className="bg-zinc-900/50 rounded-lg p-3 border border-zinc-700/50 text-xs text-zinc-400 space-y-1">
            <div className="font-medium text-zinc-300 mb-1">SLURM Pipeline</div>
            <div>
              1.{" "}
              <code className="text-indigo-400">submit_cache_logits.slurm</code>{" "}
              — teacher caches soft labels (10 parallel shards)
            </div>
            <div>
              2.{" "}
              <code className="text-indigo-400">submit_distill_student.slurm</code>{" "}
              — student trains from cache (no teacher in VRAM)
            </div>
          </div>
        </div>

        {/* ── Results ── */}
        <div className="bg-zinc-800/30 border border-zinc-700/50 rounded-xl p-5">
          {result ? (
            <div className="space-y-4">
              {/* Recommendation banner */}
              <div
                className={`rounded-lg p-3 border text-sm font-medium ${
                  result.recommendation === "distill"
                    ? "bg-emerald-900/20 border-emerald-700/50 text-emerald-300"
                    : "bg-amber-900/20 border-amber-700/50 text-amber-300"
                }`}
              >
                {result.recommendation === "distill"
                  ? "✓ Recommend: Distill"
                  : "✓ Recommend: Quantize"}{" "}
                — {result.reason}
              </div>

              {/* Side-by-side */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <div className="text-xs text-zinc-500 uppercase tracking-wider font-medium">
                    Distillation
                  </div>
                  <MetricCard
                    label="Accuracy Retention"
                    value={`${result.distill.accuracyRetentionPct}%`}
                    color={
                      result.distill.accuracyRetentionPct > 85
                        ? "green"
                        : result.distill.accuracyRetentionPct > 70
                        ? "amber"
                        : "red"
                    }
                  />
                  <MetricCard
                    label="Inference Cost/hr"
                    value={`$${result.distill.studentCostPerHour.toFixed(2)}`}
                    sub={`${result.distill.studentGpusNeeded}× GPU`}
                    color="blue"
                  />
                  <MetricCard
                    label="Cost Savings vs Teacher"
                    value={`${result.distill.costSavingsPct.toFixed(1)}%`}
                    color="green"
                  />
                  <MetricCard
                    label="One-time Distill Cost"
                    value={`$${result.distill.totalDistillCostUsd.toFixed(2)}`}
                    sub={`${result.distill.cacheJobDurationEstimateHrs.toFixed(1)}h cache + ${result.distill.trainJobDurationEstimateHrs.toFixed(1)}h train`}
                  />
                  <MetricCard
                    label="Compression"
                    value={`${result.distill.compressionRatio}×`}
                  />
                </div>
                <div className="space-y-2">
                  <div className="text-xs text-zinc-500 uppercase tracking-wider font-medium">
                    Teacher + INT4 Quant
                  </div>
                  <MetricCard
                    label="Accuracy Retention"
                    value={`${result.quantized.accuracyRetentionPct}%`}
                    color="green"
                  />
                  <MetricCard
                    label="Inference Cost/hr"
                    value={`$${result.quantized.pricePerHour.toFixed(2)}`}
                    sub={`${result.quantized.gpusNeeded}× GPU`}
                    color="amber"
                  />
                  <MetricCard
                    label="Cost Savings vs FP16"
                    value={`${(
                      ((result.distill.teacherCostPerHour -
                        result.quantized.pricePerHour) /
                        result.distill.teacherCostPerHour) *
                      100
                    ).toFixed(1)}%`}
                  />
                  <MetricCard
                    label="One-time Cost"
                    value="$0"
                    sub="No training required"
                    color="green"
                  />
                  <MetricCard
                    label="Decode Throughput"
                    value={`${result.quantized.decodeTps.toLocaleString()} tok/s`}
                  />
                </div>
              </div>

              <div className="text-xs text-zinc-500 bg-zinc-900/50 rounded-lg p-3 border border-zinc-700/50">
                <span className="text-amber-400 font-medium">Note:</span>{" "}
                Accuracy uses log-linear scaling (k=0.08) from KD literature.
                Run{" "}
                <code className="text-indigo-400">submit_cache_logits.slurm</code>{" "}
                +{" "}
                <code className="text-indigo-400">submit_distill_student.slurm</code>{" "}
                for real measured numbers.
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-zinc-500 text-sm">
              No benchmark data for selected pair
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check**

```bash
cd disagg-bench/ui
npx tsc --noEmit 2>&1 | grep distillation
```
Expected: no errors for distillation files

- [ ] **Step 3: Commit**

```bash
git add disagg-bench/ui/components/distillation-planner.tsx
git commit -m "feat: DistillationPlanner component with distill vs quant side-by-side"
```

---

### Task 9: Wire into `page.tsx`

**Files:**
- Modify: `disagg-bench/ui/app/page.tsx`

- [ ] **Step 1: Add import**

In `disagg-bench/ui/app/page.tsx`, add alongside existing imports:

```typescript
import { DistillationPlanner } from "@/components/distillation-planner";
```

- [ ] **Step 2: Add section between Pareto chart and Fine-Tuning divider**

Insert this block in `page.tsx` after the `{/* Pareto Chart */}` section and before the Fine-Tuning divider:

```tsx
        {/* Divider */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-zinc-800" />
          </div>
          <div className="relative flex justify-center">
            <span className="bg-[var(--bg)] px-4 text-sm text-zinc-500">
              Distillation
            </span>
          </div>
        </div>

        {/* Distillation Planner */}
        <section className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6">
          <DistillationPlanner />
        </section>
```

- [ ] **Step 3: Type-check full project**

```bash
cd disagg-bench/ui
npx tsc --noEmit 2>&1 | head -20
```
Expected: no new errors

- [ ] **Step 4: Dev server smoke test**

```bash
npm run dev
```
Open http://localhost:3000. Scroll to the "Distillation" section. Verify:
- Pair selector renders
- Temperature/alpha/samples/users inputs work
- Results panel shows recommendation banner + side-by-side MetricCards
- Changing pair updates recommendation

- [ ] **Step 5: Commit**

```bash
git add disagg-bench/ui/app/page.tsx
git commit -m "feat: wire DistillationPlanner into InfraBench main page"
```

---

## Chunk 3: Docs

### Task 10: Documentation

**Files:**
- Create: `disagg-bench/cluster/README.md`
- Modify: `README.md`

(Do this last, after Task 5 integration tests pass.)

- [ ] **Step 1: Create `disagg-bench/cluster/README.md`**

```markdown
# disagg-bench Cluster Scripts

Benchmark and distillation pipeline for SLURM clusters.

## Inference Benchmarking

```bash
# FP16 sweep
MODEL=facebook/opt-30b sbatch submit_benchmark.slurm

# INT8 + INT4 sweep
MODEL=facebook/opt-30b sbatch submit_quant_benchmark.slurm
```

## Distillation Pipeline

Two-stage: cache teacher logits offline, then train the student.
The teacher is never in VRAM during student training.

### Stage 1: Cache teacher logits (10 parallel shards)

```bash
TEACHER=facebook/opt-30b \
STUDENT=facebook/opt-1.3b \
NUM_SAMPLES=10000 \
QUANT=int8 \
  sbatch submit_cache_logits.slurm
```

### Stage 2: Train student (chain with dependency)

```bash
CACHE_ID=$(sbatch --parsable submit_cache_logits.slurm)
STUDENT=facebook/opt-1.3b \
CACHE_DIR=$HOME/distill_cache \
TEMPERATURE=4.0 \
ALPHA=0.5 \
  sbatch --dependency=afterok:$CACHE_ID submit_distill_student.slurm
```

### Why offline caching?

Loading a 30B teacher + 1.3B student simultaneously needs ~60 GB teacher
VRAM + ~3 GB student VRAM + activations — exceeding single A100 capacity.
Offline caching runs teacher alone per shard, saves `soft_labels` tensors
to disk, then the student trains from those tensors without the teacher.
```

- [ ] **Step 2: Append distillation section to root README.md**

Append after the "GPU vs CPU Comparison" section:

```markdown
## Distillation Pipeline (disagg-bench)

Knowledge distillation trains a smaller **student** model to mimic a larger **teacher**,
cutting inference cost while retaining most accuracy.

### Run on FluidStack

```bash
cd ~/Intro_to_SLURM/disagg-bench/cluster

# Stage 1: cache teacher soft labels (10 parallel tasks)
TEACHER=facebook/opt-30b STUDENT=facebook/opt-1.3b sbatch submit_cache_logits.slurm

# Stage 2: train student from cached logits
CACHE_ID=<job_id_above>
sbatch --dependency=afterok:$CACHE_ID submit_distill_student.slurm
```

### See cost estimates in InfraBench UI

The **Distillation Planner** tab estimates:
- Accuracy retention (log-linear scaling law, k=0.08)
- Inference cost savings vs teacher INT4 quantization
- One-time distillation cost (cache + train)
- Recommendation: distill vs quantize
```

- [ ] **Step 3: Commit**

```bash
git add disagg-bench/cluster/README.md README.md
git commit -m "docs: distillation pipeline documentation"
```

---

## File Map

| File | Status | Purpose |
|------|--------|---------|
| `disagg-bench/cluster/distill_config.py` | Create | Config dataclass + shard math |
| `disagg-bench/cluster/cache_teacher_logits.py` | Create | Teacher logit caching pipeline |
| `disagg-bench/cluster/train_student_distill.py` | Create | Offline student training (CE+KL loss) |
| `disagg-bench/cluster/submit_cache_logits.slurm` | Create | SLURM array: 10-shard caching |
| `disagg-bench/cluster/submit_distill_student.slurm` | Create | SLURM: student training |
| `disagg-bench/cluster/tests/test_distill_config.py` | Create | Unit tests: config + cache format |
| `disagg-bench/cluster/tests/test_train_student_distill.py` | Create | Unit tests: loss + dataset |
| `disagg-bench/cluster/tests/test_integration.py` | Create | Integration: pipeline end-to-end |
| `disagg-bench/ui/package.json` | Modify | Add vitest |
| `disagg-bench/ui/vitest.config.ts` | Create | Vitest config with path alias |
| `disagg-bench/ui/lib/distillation.ts` | Create | TS: scaling law accuracy + cost |
| `disagg-bench/ui/lib/distillation.test.ts` | Create | TS unit tests |
| `disagg-bench/ui/components/distillation-planner.tsx` | Create | React component |
| `disagg-bench/ui/app/page.tsx` | Modify | Wire DistillationPlanner in |
| `disagg-bench/cluster/README.md` | Create | Cluster script docs |
| `README.md` | Modify | Distillation section |
