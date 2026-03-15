# InfraBench Cluster — LLM Distillation Pipeline

Run knowledge distillation on a SLURM cluster: cache teacher soft-labels in parallel,
then train a smaller student model from those cached tensors.

## Why Offline Distillation?

The core problem: a 66B-parameter teacher model and a 1.3B student don't fit in VRAM
at the same time on most nodes. Offline distillation solves this by separating the
two phases completely:

1. **Cache phase** — run the teacher alone, save `softmax(logits / T)` to disk
2. **Train phase** — student trains from cached tensors; teacher never touches VRAM

This lets you use a large quantized teacher (INT4 = ~33 GB for 66B params) without
constraining the student's training environment.

## Files

| File | Purpose |
|------|---------|
| `distill_config.py` | Shared config dataclass; shard math, cache path generation |
| `cache_teacher_logits.py` | Teacher forward pass → `.pt` shard files on disk |
| `train_student_distill.py` | Student training from cached logits (CE + KL loss) |
| `submit_cache_logits.slurm` | SLURM array job — 10 shards in parallel |
| `submit_distill_student.slurm` | SLURM training job — runs after cache array completes |
| `tests/` | Unit + integration tests (pytest) |

## Quick Start

### Step 1 — Cache teacher soft-labels (parallel, 10 shards)

```bash
CACHE_ID=$(sbatch --parsable submit_cache_logits.slurm)
echo "Cache job: $CACHE_ID"
```

Each SLURM array task processes `total_samples / 10` examples and writes
`output/shard_NNNN_of_0010.pt`.

### Step 2 — Train student (runs after all shards finish)

```bash
sbatch --dependency=afterok:$CACHE_ID submit_distill_student.slurm
```

The `--dependency=afterok` flag ensures training only starts after every
cache shard succeeds. If any shard fails, the training job is cancelled
automatically.

### Monitor

```bash
squeue -u $USER                        # running jobs
tail -f logs/cache_*.out               # stream cache output
tail -f logs/distill_student_*.out     # stream training output
```

## Local Testing (no SLURM)

```bash
cd disagg-bench/cluster

# Run single cache shard
python3 cache_teacher_logits.py \
  --teacher-model facebook/opt-1.3b \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --num-samples 100 \
  --shard-id 0 \
  --num-shards 1 \
  --output-dir /tmp/distill_cache

# Train student from cache
python3 train_student_distill.py \
  --student-model facebook/opt-125m \
  --cache-dir /tmp/distill_cache \
  --output-dir /tmp/distill_output \
  --num-epochs 1
```

## Run Tests

```bash
pip install pytest torch
python3 -m pytest tests/ -v
```

Expected output: 13 tests passing.

## How the Loss Works

Student training minimizes a combined loss:

```
loss = α · CE(student_logits, labels) + (1 - α) · T² · KL(student_soft || teacher_soft)
```

Where:
- **CE** = standard cross-entropy against ground-truth labels
- **KL** = KL divergence between student and teacher soft distributions
- **T** = temperature (default 4.0) — higher T spreads the teacher's probability mass,
  revealing more signal in near-zero probabilities ("dark knowledge")
- **T²** = re-scaling factor so the KL magnitude matches CE at any temperature
- **α** = balance weight (default 0.5) — 0 = pure distillation, 1 = pure CE

Reference: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015).

## Config Reference

`distill_config.py` — `DistillConfig` dataclass:

| Field | Default | Description |
|-------|---------|-------------|
| `teacher_model` | — | HuggingFace model ID or local path |
| `student_model` | — | HuggingFace model ID or local path |
| `dataset_name` | `wikitext` | HuggingFace dataset name |
| `dataset_config` | `wikitext-2-raw-v1` | Dataset configuration |
| `max_seq_length` | `512` | Truncation length for tokenizer |
| `temperature` | `4.0` | Soft-label temperature (τ) |
| `alpha` | `0.5` | CE weight in combined loss |
| `num_samples` | `10000` | Total dataset samples across all shards |
| `num_shards` | `10` | Number of parallel cache shards |
| `shard_id` | `0` | This shard's index (0-indexed) |
| `output_dir` | — | Directory for cache `.pt` files |

## Cache File Format

Each shard writes a `.pt` file loadable with `torch.load(..., weights_only=False)`:

```python
{
    "input_ids": Tensor(N, seq_len),          # tokenized inputs
    "soft_labels": Tensor(N, seq_len, vocab), # softmax(logits / T)
    "temperature": float,                      # T used during caching
    "model": str,                              # teacher model ID
    "vocab_size": int,                         # vocabulary size
}
```

## Accuracy vs. Cost Trade-Off

The UI's Distillation Planner tab shows estimated accuracy retention before you run
anything. The estimate uses the empirical log-linear scaling law from DistilBERT and
TinyBERT literature:

```
retention = 1 - 0.08 * ln(teacher_params / student_params)
```

Example: 66B teacher → 1.3B student ≈ 60% accuracy retention at ~98% cost savings.
