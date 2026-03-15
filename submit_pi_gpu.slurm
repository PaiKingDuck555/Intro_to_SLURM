#!/bin/bash
#SBATCH --job-name=monte_carlo_pi_gpu
#SBATCH --output=logs/pi_gpu_%j.out
#SBATCH --error=logs/pi_gpu_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=priority

# ──────────────────────────────────────────────
#  Monte Carlo Pi  –  GPU vs CPU benchmark
#
#  10 billion darts, chunked to avoid OOM.
#  CPU run will take a while — GPU should crush it.
# ──────────────────────────────────────────────

TOTAL_DARTS=100000000000

export PATH=$HOME/.local/bin:$PATH

echo "============================================"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  GPUs    : $CUDA_VISIBLE_DEVICES"
echo "  Darts   : $TOTAL_DARTS"
echo "  Started : $(date)"
echo "============================================"

nvidia-smi

mkdir -p logs results

# Run from repo root: cd Intro_to_SLURM && sbatch submit_pi_gpu.slurm
python3 monte_carlo_pi_gpu.py \
    --total-darts $TOTAL_DARTS \
    --chunked \
    --skip-cpu \
    --output-dir results

echo "Finished: $(date)"
