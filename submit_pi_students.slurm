#!/bin/bash
#SBATCH --job-name=pi_students
#SBATCH --output=logs/pi_students_%A_%a.out
#SBATCH --error=logs/pi_students_%A_%a.err
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --partition=priority

# ──────────────────────────────────────────────
#  Monte Carlo Pi  –  Distillation STUDENTS (array)
#
#  10 student runs in parallel; each does 5M darts.
#  Run after submit_pi_teacher.slurm; then combine with:
#    python3 combine_results.py --results-dir results --mode distill
# ──────────────────────────────────────────────

NUM_TASKS=10
STUDENT_DARTS="${STUDENT_DARTS:-5000000}"
RESULTS_DIR="${RESULTS_DIR:-results}"

export PATH=$HOME/.local/bin:$PATH

echo "============================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Array Task : $SLURM_ARRAY_TASK_ID / $NUM_TASKS"
echo "  Node       : $(hostname)"
echo "  Darts/task : $STUDENT_DARTS"
echo "  Started    : $(date)"
echo "============================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p logs "$RESULTS_DIR"

# Run from repo root: cd Intro_to_SLURM && sbatch submit_pi_students.slurm
CHUNKED_ARGS=""
if [ -n "${CHUNKED}" ]; then CHUNKED_ARGS="--chunked"; fi

python3 distill_pi.py \
    --mode student \
    --student-darts "$STUDENT_DARTS" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --num-tasks "$NUM_TASKS" \
    --output-dir "$RESULTS_DIR" \
    $CHUNKED_ARGS

echo "Finished: $(date)"
