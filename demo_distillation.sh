#!/usr/bin/env bash
# Run the full distillation pipeline locally (no SLURM, no GPU required).
# Uses small dart counts so it finishes in seconds. Perfect for demos and judges.
set -e
cd "$(dirname "$0")"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR" logs

echo "============================================"
echo "  Monte Carlo Pi — Distillation demo"
echo "  (teacher + students → combined estimate)"
echo "============================================"

# Teacher: 50k darts
echo ""
echo "[1/3] Teacher run (50,000 darts)..."
python3 distill_pi.py --mode teacher --teacher-darts 50000 --output-dir "$RESULTS_DIR"

# Students: 4 × 25k darts
echo ""
echo "[2/3] Student runs (4 × 25,000 darts)..."
for i in 0 1 2 3; do
  python3 distill_pi.py --mode student --student-darts 25000 --task-id $i --num-tasks 4 --output-dir "$RESULTS_DIR"
done

# Combine
echo ""
echo "[3/3] Combining teacher + students..."
python3 combine_results.py --results-dir "$RESULTS_DIR" --mode distill

echo ""
echo "Done. Check $RESULTS_DIR/distilled_summary.json for the full report."
