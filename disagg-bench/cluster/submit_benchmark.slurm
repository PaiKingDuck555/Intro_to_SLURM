#!/bin/bash
#SBATCH --job-name=disagg_bench
#SBATCH --output=logs/bench_%A_%a.out
#SBATCH --error=logs/bench_%A_%a.err
#SBATCH --array=0-14
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=priority

# ──────────────────────────────────────────────
#  Disaggregated Inference Benchmark Sweep
#
#  Tests 15 configurations (5 batch × 3 seq len)
#  for a given model. Each array task = 1 config.
# ──────────────────────────────────────────────

export PATH=$HOME/.local/bin:$PATH

MODEL="${MODEL:-facebook/opt-1.3b}"
DTYPE="${DTYPE:-float16}"

BATCH_SIZES=(1 4 8 32 64)
SEQ_LENGTHS=(128 512 2048)

NUM_SEQ=${#SEQ_LENGTHS[@]}
BATCH_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEQ))
SEQ_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEQ))

BATCH=${BATCH_SIZES[$BATCH_IDX]}
SEQ=${SEQ_LENGTHS[$SEQ_IDX]}

echo "============================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Array Task : $SLURM_ARRAY_TASK_ID / 15"
echo "  Node       : $(hostname)"
echo "  Model      : $MODEL"
echo "  Batch Size : $BATCH"
echo "  Seq Length : $SEQ"
echo "  Dtype      : $DTYPE"
echo "  Started    : $(date)"
echo "============================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p logs results

python3 ~/disagg-bench/cluster/benchmark_inference.py \
    --model "$MODEL" \
    --batch-size "$BATCH" \
    --seq-length "$SEQ" \
    --decode-tokens 100 \
    --dtype "$DTYPE" \
    --output-dir ~/disagg-bench/results

echo "Finished: $(date)"
