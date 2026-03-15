#!/usr/bin/env python3
"""
Monte Carlo Pi distillation: teacher + student runs for optimized GPU usage.

Distillation workflow:
  1. Teacher: one high-dart GPU run → reference estimate + variance (results/teacher.json).
  2. Students: many smaller GPU runs (SLURM array) → results/student_<id>.json.
  3. Combine: inverse-variance weighted π from teacher + students → results/distilled_summary.json.

Benefits:
  - Teacher warms GPU once; students are small and parallelizable (array jobs).
  - Same total darts, better GPU utilization and optional variance/CI reporting.
  - Students can use chunked mode on low-VRAM nodes.

Usage:
  # Teacher (single job)
  python3 distill_pi.py --mode teacher --teacher-darts 50000000 --output-dir results

  # Student (per array task)
  python3 distill_pi.py --mode student --student-darts 5000000 --task-id 0 --num-tasks 10

  # Combine after teacher + all students finish
  python3 distill_pi.py --mode combine --results-dir results
"""

import argparse
import glob
import json
import math
import os
import sys
import time

# Allow running from repo root or from this directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Reuse GPU/CPU estimation from the main GPU script
try:
    from monte_carlo_pi_gpu import (
        GPU_AVAILABLE,
        estimate_pi_gpu,
        estimate_pi_gpu_chunked,
        estimate_pi_cpu,
    )
    if GPU_AVAILABLE:
        import cupy as cp
except ImportError:
    GPU_AVAILABLE = False
    estimate_pi_gpu = estimate_pi_gpu_chunked = estimate_pi_cpu = None

import numpy as np


# Theoretical variance of π̂ for n darts: var(π̂) ≈ π(4-π)/n ≈ 2.70/n
PI_VAR_COEF = (np.pi * (4 - np.pi))  # ≈ 2.696


def run_teacher(num_darts: int, seed: int, chunked: bool, output_dir: str) -> dict:
    """One high-dart GPU run; write teacher.json."""
    start = time.perf_counter()
    use_gpu = GPU_AVAILABLE
    if use_gpu:
        try:
            if chunked:
                result = estimate_pi_gpu_chunked(num_darts, seed)
            else:
                result = estimate_pi_gpu(num_darts, seed)
            if GPU_AVAILABLE:
                cp.cuda.Device(0).synchronize()
        except Exception as e:
            print(f"GPU failed: {e}, falling back to CPU", file=sys.stderr)
            use_gpu = False
    if not use_gpu:
        result = estimate_pi_cpu(num_darts, seed)

    result["role"] = "teacher"
    result["variance_estimate"] = PI_VAR_COEF / num_darts
    result["std_error"] = math.sqrt(result["variance_estimate"])
    result["elapsed_seconds"] = round(time.perf_counter() - start, 4)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "teacher.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Teacher: π ≈ {result['pi_estimate']:.10f}  darts={num_darts:,}  → {path}")
    return result


def run_student(
    num_darts: int,
    task_id: int,
    num_tasks: int,
    seed_base: int,
    chunked: bool,
    output_dir: str,
) -> dict:
    """One student GPU run; write student_<task_id>.json."""
    start = time.perf_counter()
    seed = seed_base + 1000 + task_id  # distinct from teacher
    use_gpu = GPU_AVAILABLE
    if use_gpu:
        try:
            if chunked:
                result = estimate_pi_gpu_chunked(num_darts, seed)
            else:
                result = estimate_pi_gpu(num_darts, seed)
            if GPU_AVAILABLE:
                cp.cuda.Device(0).synchronize()
        except Exception as e:
            print(f"GPU failed: {e}, falling back to CPU", file=sys.stderr)
            use_gpu = False
    if not use_gpu:
        result = estimate_pi_cpu(num_darts, seed)

    result["task_id"] = task_id
    result["num_tasks"] = num_tasks
    result["role"] = "student"
    result["variance_estimate"] = PI_VAR_COEF / num_darts
    result["elapsed_seconds"] = round(time.perf_counter() - start, 4)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"student_{task_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Student {task_id}: π ≈ {result['pi_estimate']:.10f}  darts={num_darts:,}  → {path}")
    return result


def combine_distilled(results_dir: str) -> dict:
    """
    Combine teacher.json + student_*.json into one inverse-variance weighted π estimate.
    Optimal weight for each run is proportional to n (darts); equivalent to sum(hits)/sum(darts).
    """
    teacher_path = os.path.join(results_dir, "teacher.json")
    student_files = sorted(glob.glob(os.path.join(results_dir, "student_*.json")))

    total_darts = 0
    total_hits = 0
    runs = []

    if os.path.isfile(teacher_path):
        with open(teacher_path) as f:
            t = json.load(f)
        total_darts += t["darts"]
        total_hits += t["hits"]
        runs.append({"role": "teacher", **t})
        print(f"  Teacher: darts={t['darts']:>12,}  hits={t['hits']:>12,}  π≈{t['pi_estimate']:.8f}")

    for path in student_files:
        with open(path) as f:
            s = json.load(f)
        total_darts += s["darts"]
        total_hits += s["hits"]
        runs.append({"role": "student", "task_id": s.get("task_id"), **s})
        print(f"  Student {s.get('task_id', '?'):>2}: darts={s['darts']:>12,}  hits={s['hits']:>12,}  π≈{s['pi_estimate']:.8f}")

    if total_darts == 0:
        print("No teacher or student result files found.", file=sys.stderr)
        sys.exit(1)

    pi_distilled = 4.0 * total_hits / total_darts
    error_abs = abs(pi_distilled - math.pi)
    variance = PI_VAR_COEF / total_darts
    std_error = math.sqrt(variance)
    # Approximate 95% CI (normal approximation)
    ci_95_lo = pi_distilled - 1.96 * std_error
    ci_95_hi = pi_distilled + 1.96 * std_error

    summary = {
        "total_darts": total_darts,
        "total_hits": total_hits,
        "pi_estimate": pi_distilled,
        "actual_pi": math.pi,
        "error": error_abs,
        "variance_estimate": variance,
        "std_error": std_error,
        "ci_95": [ci_95_lo, ci_95_hi],
        "num_runs": len(runs),
        "num_teacher": 1 if os.path.isfile(teacher_path) else 0,
        "num_students": len(student_files),
    }

    print("\n" + "=" * 56)
    print("  Distilled estimate (teacher + students)")
    print(f"  Total darts : {total_darts:>14,}")
    print(f"  Total hits  : {total_hits:>14,}")
    print(f"  π estimate  : {pi_distilled:.12f}")
    print(f"  Actual π    : {math.pi:.12f}")
    print(f"  Error       : {error_abs:.2e}")
    print(f"  Std error   : {std_error:.2e}")
    print(f"  95% CI      : [{ci_95_lo:.6f}, {ci_95_hi:.6f}]")
    print("=" * 56)

    out_path = os.path.join(results_dir, "distilled_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDistilled summary written to {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Pi distillation: teacher + student GPU runs"
    )
    parser.add_argument("--mode", choices=["teacher", "student", "combine"], required=True)
    parser.add_argument("--output-dir", "--results-dir", dest="results_dir", default="results")
    # Teacher
    parser.add_argument("--teacher-darts", type=int, default=50_000_000)
    parser.add_argument("--teacher-seed", type=int, default=42)
    # Student (array)
    parser.add_argument("--student-darts", type=int, default=5_000_000)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=42)
    # GPU
    parser.add_argument("--chunked", action="store_true", help="Use chunked GPU mode (low VRAM)")
    args = parser.parse_args()

    if args.mode == "teacher":
        start = time.perf_counter()
        run_teacher(
            args.teacher_darts,
            args.teacher_seed,
            args.chunked,
            args.results_dir,
        )
        elapsed = time.perf_counter() - start
        print(f"Teacher run finished in {elapsed:.2f}s")

    elif args.mode == "student":
        start = time.perf_counter()
        run_student(
            args.student_darts,
            args.task_id,
            args.num_tasks,
            args.seed_base,
            args.chunked,
            args.results_dir,
        )
        elapsed = time.perf_counter() - start
        print(f"Student {args.task_id} finished in {elapsed:.2f}s")

    else:  # combine
        combine_distilled(args.results_dir)


if __name__ == "__main__":
    main()
