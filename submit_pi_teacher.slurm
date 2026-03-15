#!/bin/bash
#SBATCH --job-name=pi_teacher
#SBATCH --output=logs/pi_teacher_%j.out
#SBATCH --error=logs/pi_teacher_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=priority

# ──────────────────────────────────────────────
#  Monte Carlo Pi  –  Distillation TEACHER
#
#  One high-dart GPU run to produce a reference estimate.
#  Run first; then submit_pi_students.slurm (array); then combine.
#
#  Usage:
#    sbatch submit_pi_teacher.slurm
#    # after it finishes:
#    sbatch submit_pi_students.slurm
#    # after array finishes:
#    python3 combine_results.py --results-dir results --mode distill
# ──────────────────────────────────────────────

TEACHER_DARTS="${TEACHER_DARTS:-50000000}"
RESULTS_DIR="${RESULTS_DIR:-results}"

export PATH=$HOME/.local/bin:$PATH

echo "============================================"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $(hostname)"
echo "  GPUs     : $CUDA_VISIBLE_DEVICES"
echo "  Darts    : $TEACHER_DARTS (teacher)"
echo "  Started  : $(date)"
echo "============================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p logs "$RESULTS_DIR"

# Run from repo root: cd Intro_to_SLURM && sbatch submit_pi_teacher.slurm
python3 distill_pi.py \
    --mode teacher \
    --teacher-darts "$TEACHER_DARTS" \
    --output-dir "$RESULTS_DIR"

echo "Finished: $(date)"
