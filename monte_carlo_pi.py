#!/usr/bin/env python3
"""
Monte Carlo estimation of Pi.

Geometry:
  - Square: 2π × 2π centered at origin → x,y ∈ [-π, π]
  - Circle: radius π centered at origin
  - Area ratio: π·r² / (2r)² = π/4  →  Pi ≈ 4 × (hits / total)

Usage:
  python monte_carlo_pi.py --total-darts 100000000 --task-id 0 --num-tasks 1
"""

import argparse
import json
import os
import time

import numpy as np


def estimate_pi(num_darts: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    half_side = np.pi  # square spans [-π, π] in both axes
    radius = np.pi

    chunk = 10_000_000
    hits = 0
    remaining = num_darts

    while remaining > 0:
        n = min(chunk, remaining)
        x = rng.uniform(-half_side, half_side, size=n)
        y = rng.uniform(-half_side, half_side, size=n)
        hits += int(np.sum(x * x + y * y <= radius * radius))
        remaining -= n

    pi_estimate = 4.0 * hits / num_darts
    return {
        "darts": num_darts,
        "hits": int(hits),
        "pi_estimate": pi_estimate,
        "error": abs(pi_estimate - np.pi),
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Pi estimation")
    parser.add_argument("--total-darts", type=int, default=100_000_000)
    parser.add_argument("--task-id", type=int, default=0,
                        help="SLURM array task index (0-based)")
    parser.add_argument("--num-tasks", type=int, default=1,
                        help="Total number of parallel SLURM tasks")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to write per-task JSON results")
    args = parser.parse_args()

    darts_per_task = args.total_darts // args.num_tasks
    leftover = args.total_darts % args.num_tasks
    if args.task_id < leftover:
        darts_per_task += 1

    seed = 42 + args.task_id

    print(f"[Task {args.task_id}/{args.num_tasks}] "
          f"Throwing {darts_per_task:,} darts (seed={seed}) ...")

    start = time.perf_counter()
    result = estimate_pi(darts_per_task, seed)
    elapsed = time.perf_counter() - start

    result["task_id"] = args.task_id
    result["elapsed_seconds"] = round(elapsed, 3)

    print(f"[Task {args.task_id}] Pi ≈ {result['pi_estimate']:.10f}  "
          f"(error={result['error']:.2e}, {elapsed:.2f}s)")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"task_{args.task_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Task {args.task_id}] Results written to {out_path}")


if __name__ == "__main__":
    main()
