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
