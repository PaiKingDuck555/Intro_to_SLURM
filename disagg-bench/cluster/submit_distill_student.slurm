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
