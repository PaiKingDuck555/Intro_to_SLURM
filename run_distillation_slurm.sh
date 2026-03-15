#!/usr/bin/env bash
# Submit teacher then students (with dependency) so the full distillation runs on the cluster.
# After both finish, run: python3 combine_results.py --results-dir results --mode distill
set -e
cd "$(dirname "$0")"
mkdir -p logs results

echo "Submitting teacher job..."
TEACHER_JOB=$(sbatch --parsable submit_pi_teacher.slurm)
echo "  Teacher job ID: $TEACHER_JOB"

echo "Submitting student array (runs after teacher completes)..."
STUDENT_JOB=$(sbatch --parsable --dependency=afterok:$TEACHER_JOB submit_pi_students.slurm)
echo "  Student array job ID: $STUDENT_JOB"

echo ""
echo "============================================"
echo "  Distillation pipeline submitted"
echo "============================================"
echo "  Teacher : $TEACHER_JOB"
echo "  Students: $STUDENT_JOB (depends on teacher)"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  When both finish, run:"
echo "    python3 combine_results.py --results-dir results --mode distill"
echo "============================================"
