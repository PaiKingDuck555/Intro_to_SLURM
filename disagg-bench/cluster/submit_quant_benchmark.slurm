#!/bin/bash
#SBATCH --job-name=quant_bench
#SBATCH --output=logs/quant_%A_%a.out
#SBATCH --error=logs/quant_%A_%a.err
#SBATCH --array=0-29
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=priority

# ──────────────────────────────────────────────
#  Quantization Benchmark Sweep
#
#  Sweeps: 2 quant levels × 5 batch × 3 seq = 30 configs
#  per model. Each array task = 1 config.
#
#  Usage:
#    MODEL=facebook/opt-30b sbatch submit_quant_benchmark.slurm
# ──────────────────────────────────────────────

export PATH=$HOME/.local/bin:$PATH

MODEL="${MODEL:-facebook/opt-30b}"

QUANT_LEVELS=(int8 int4)
BATCH_SIZES=(1 4 8 32 64)
SEQ_LENGTHS=(128 512 2048)

NUM_SEQ=${#SEQ_LENGTHS[@]}
NUM_BATCH=${#BATCH_SIZES[@]}
CONFIGS_PER_QUANT=$((NUM_BATCH * NUM_SEQ))

QUANT_IDX=$((SLURM_ARRAY_TASK_ID / CONFIGS_PER_QUANT))
REMAINING=$((SLURM_ARRAY_TASK_ID % CONFIGS_PER_QUANT))
BATCH_IDX=$((REMAINING / NUM_SEQ))
SEQ_IDX=$((REMAINING % NUM_SEQ))

QUANT=${QUANT_LEVELS[$QUANT_IDX]}
BATCH=${BATCH_SIZES[$BATCH_IDX]}
SEQ=${SEQ_LENGTHS[$SEQ_IDX]}

echo "============================================"
echo "  Job ID       : $SLURM_JOB_ID"
echo "  Array Task   : $SLURM_ARRAY_TASK_ID / 30"
echo "  Node         : $(hostname)"
echo "  Model        : $MODEL"
echo "  Quantization : $QUANT"
echo "  Batch Size   : $BATCH"
echo "  Seq Length   : $SEQ"
echo "  Started      : $(date)"
echo "============================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p logs results

python3 ~/disagg-bench/cluster/benchmark_inference.py \
    --model "$MODEL" \
    --batch-size "$BATCH" \
    --seq-length "$SEQ" \
    --decode-tokens 100 \
    --dtype float16 \
    --quantization "$QUANT" \
    --output-dir ~/disagg-bench/results

echo "Finished: $(date)"
