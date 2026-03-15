#!/bin/bash
#SBATCH --job-name=monte_carlo_pi
#SBATCH --output=logs/pi_%A_%a.out
#SBATCH --error=logs/pi_%A_%a.err
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu

# ──────────────────────────────────────────────
#  Monte Carlo Pi  –  SLURM Array Job
#
#  Splits 100,000,000 darts across 10 tasks.
#  Each task throws 10M darts independently,
#  then combine_results.py merges them.
# ──────────────────────────────────────────────

NUM_TASKS=10
TOTAL_DARTS=100000000

echo "============================================"
echo "  Job ID       : $SLURM_JOB_ID"
echo "  Array Task   : $SLURM_ARRAY_TASK_ID / $NUM_TASKS"
echo "  Node         : $(hostname)"
echo "  Started      : $(date)"
echo "============================================"

mkdir -p logs results

python3 monte_carlo_pi.py \
    --total-darts $TOTAL_DARTS \
    --task-id $SLURM_ARRAY_TASK_ID \
    --num-tasks $NUM_TASKS \
    --output-dir results

echo "Finished: $(date)"
