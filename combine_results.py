#!/usr/bin/env python3
"""
Aggregate per-task JSON results into a single Pi estimate.

Modes:
  array  (default): Combine task_*.json from CPU SLURM array → summary.json
  distill:           Combine teacher.json + student_*.json from GPU distillation → distilled_summary.json

Run after all SLURM array tasks complete:
    python3 combine_results.py --results-dir results

Run after distillation teacher + student jobs:
    python3 combine_results.py --results-dir results --mode distill
"""

import argparse
import glob
import json
import math
import os
import sys


# Theoretical variance of π̂ for n darts: var(π̂) ≈ π(4-π)/n
PI_VAR_COEF = math.pi * (4 - math.pi)


def combine_distill(results_dir: str) -> None:
    """Combine teacher.json + student_*.json into distilled_summary.json."""
    teacher_path = os.path.join(results_dir, "teacher.json")
    student_files = sorted(glob.glob(os.path.join(results_dir, "student_*.json")))

    total_darts = 0
    total_hits = 0

    if os.path.isfile(teacher_path):
        with open(teacher_path) as f:
            t = json.load(f)
        total_darts += t["darts"]
        total_hits += t["hits"]
        el = t.get("elapsed_seconds")
        el_str = f"  ({el:.1f}s)" if el is not None else ""
        print(f"  Teacher: darts={t['darts']:>12,}  hits={t['hits']:>12,}  π≈{t['pi_estimate']:.8f}{el_str}")

    for path in student_files:
        with open(path) as f:
            s = json.load(f)
        total_darts += s["darts"]
        total_hits += s["hits"]
        tid = s.get("task_id", "?")
        el = s.get("elapsed_seconds")
        el_str = f"  ({el:.1f}s)" if el is not None else ""
        print(f"  Student {tid:>2}: darts={s['darts']:>12,}  hits={s['hits']:>12,}  π≈{s['pi_estimate']:.8f}{el_str}")

    if total_darts == 0:
        print("No teacher or student result files found.", file=sys.stderr)
        sys.exit(1)

    pi_combined = 4.0 * total_hits / total_darts
    error = abs(pi_combined - math.pi)
    variance = PI_VAR_COEF / total_darts
    std_error = math.sqrt(variance)
    ci_95_lo = pi_combined - 1.96 * std_error
    ci_95_hi = pi_combined + 1.96 * std_error

    print("\n" + "=" * 56)
    print("  Distilled estimate (teacher + students)")
    print(f"  Total darts : {total_darts:>14,}")
    print(f"  Total hits  : {total_hits:>14,}")
    print(f"  Pi estimate : {pi_combined:.12f}")
    print(f"  Actual Pi   : {math.pi:.12f}")
    print(f"  Error       : {error:.2e}")
    print(f"  Std error   : {std_error:.2e}")
    print(f"  95% CI      : [{ci_95_lo:.6f}, {ci_95_hi:.6f}]")
    print("=" * 56)

    summary = {
        "total_darts": total_darts,
        "total_hits": total_hits,
        "pi_estimate": pi_combined,
        "actual_pi": math.pi,
        "error": error,
        "variance_estimate": variance,
        "std_error": std_error,
        "ci_95": [ci_95_lo, ci_95_hi],
        "num_teacher": 1 if os.path.isfile(teacher_path) else 0,
        "num_students": len(student_files),
    }
    summary_path = os.path.join(results_dir, "distilled_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    print(
        f"\n  → Distilled π = {pi_combined:.6f}  "
        f"(95% CI [{ci_95_lo:.4f}, {ci_95_hi:.4f}]). "
        f"True π = {math.pi:.6f}."
    )


def main():
    parser = argparse.ArgumentParser(description="Combine Monte Carlo results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--mode",
        choices=["array", "distill"],
        default="array",
        help="array = task_*.json (CPU); distill = teacher + student_*.json (GPU)",
    )
    args = parser.parse_args()

    if args.mode == "distill":
        combine_distill(args.results_dir)
        return

    files = sorted(glob.glob(f"{args.results_dir}/task_*.json"))
    if not files:
        print(f"No result files found in {args.results_dir}/", file=sys.stderr)
        sys.exit(1)

    total_darts = 0
    total_hits = 0
    task_estimates = []

    for path in files:
        with open(path) as f:
            data = json.load(f)
        total_darts += data["darts"]
        total_hits += data["hits"]
        task_estimates.append(data)
        print(f"  Task {data['task_id']:>2}: "
              f"darts={data['darts']:>12,}  "
              f"hits={data['hits']:>12,}  "
              f"pi≈{data['pi_estimate']:.8f}  "
              f"({data['elapsed_seconds']:.1f}s)")

    pi_combined = 4.0 * total_hits / total_darts
    error = abs(pi_combined - math.pi)

    print("\n" + "=" * 56)
    print(f"  Combined estimate from {len(files)} tasks")
    print(f"  Total darts : {total_darts:>14,}")
    print(f"  Total hits  : {total_hits:>14,}")
    print(f"  Pi estimate : {pi_combined:.12f}")
    print(f"  Actual Pi   : {math.pi:.12f}")
    print(f"  Error       : {error:.2e}")
    print("=" * 56)

    summary = {
        "total_darts": total_darts,
        "total_hits": total_hits,
        "pi_estimate": pi_combined,
        "actual_pi": math.pi,
        "error": error,
        "num_tasks": len(files),
    }
    summary_path = f"{args.results_dir}/summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
