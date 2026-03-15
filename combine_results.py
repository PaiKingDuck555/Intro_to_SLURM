#!/usr/bin/env python3
"""
Aggregate per-task JSON results into a single Pi estimate.

Run after all SLURM array tasks complete:
    python3 combine_results.py --results-dir results
"""

import argparse
import glob
import json
import math
import sys


def main():
    parser = argparse.ArgumentParser(description="Combine Monte Carlo results")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

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
