#!/usr/bin/env python3
"""
GPU-accelerated Monte Carlo estimation of Pi using CuPy.

Same geometry as the CPU version:
  - Square: 2π × 2π centered at origin → x,y ∈ [-π, π]
  - Circle: radius π centered at origin
  - Pi ≈ 4 × (hits / total)

CuPy mirrors numpy's API but runs everything on the GPU. The entire
100M dart computation happens in GPU memory — no data round-trips
to the CPU until the final count.

Usage:
  python monte_carlo_pi_gpu.py --total-darts 100000000
"""

import argparse
import json
import os
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

import numpy as np


def estimate_pi_gpu(num_darts: int, seed: int) -> dict:
    """Run the full simulation on the GPU."""
    rng = cp.random.default_rng(seed)

    half_side = cp.float64(np.pi)
    radius_sq = cp.float64(np.pi ** 2)

    # Generate all darts at once in GPU VRAM.
    # 100M float64 pairs ≈ 1.6 GB — fits comfortably in an A100's 80 GB.
    x = rng.uniform(-half_side, half_side, size=num_darts)
    y = rng.uniform(-half_side, half_side, size=num_darts)

    inside = x * x + y * y <= radius_sq
    hits = int(cp.sum(inside))           # single transfer: GPU → CPU

    pi_estimate = 4.0 * hits / num_darts
    return {
        "darts": num_darts,
        "hits": hits,
        "pi_estimate": pi_estimate,
        "error": abs(pi_estimate - np.pi),
        "seed": seed,
        "device": "gpu",
    }


def estimate_pi_gpu_chunked(num_darts: int, seed: int,
                            chunk: int = 50_000_000) -> dict:
    """
    Chunked GPU version for GPUs with limited VRAM.
    Processes darts in batches to avoid out-of-memory errors.
    """
    rng = cp.random.default_rng(seed)

    half_side = cp.float64(np.pi)
    radius_sq = cp.float64(np.pi ** 2)

    hits = 0
    remaining = num_darts

    while remaining > 0:
        n = min(chunk, remaining)
        x = rng.uniform(-half_side, half_side, size=n)
        y = rng.uniform(-half_side, half_side, size=n)
        hits += int(cp.sum(x * x + y * y <= radius_sq))
        remaining -= n

    pi_estimate = 4.0 * hits / num_darts
    return {
        "darts": num_darts,
        "hits": hits,
        "pi_estimate": pi_estimate,
        "error": abs(pi_estimate - np.pi),
        "seed": seed,
        "device": "gpu",
    }


def estimate_pi_cpu(num_darts: int, seed: int) -> dict:
    """Fallback: run on CPU with numpy (same logic as monte_carlo_pi.py)."""
    rng = np.random.default_rng(seed)

    half_side = np.pi
    radius_sq = np.pi ** 2

    chunk = 10_000_000
    hits = 0
    remaining = num_darts

    while remaining > 0:
        n = min(chunk, remaining)
        x = rng.uniform(-half_side, half_side, size=n)
        y = rng.uniform(-half_side, half_side, size=n)
        hits += int(np.sum(x * x + y * y <= radius_sq))
        remaining -= n

    pi_estimate = 4.0 * hits / num_darts
    return {
        "darts": num_darts,
        "hits": hits,
        "pi_estimate": pi_estimate,
        "error": abs(pi_estimate - np.pi),
        "seed": seed,
        "device": "cpu",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Pi — GPU (CuPy) vs CPU (numpy) comparison")
    parser.add_argument("--total-darts", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true",
                        help="Skip GPU even if available")
    parser.add_argument("--chunked", action="store_true",
                        help="Use chunked GPU mode (for GPUs with < 4GB VRAM)")
    parser.add_argument("--skip-cpu", action="store_true",
                        help="Skip CPU run (useful for large dart counts)")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    use_gpu = GPU_AVAILABLE and not args.force_cpu

    # ── GPU run ──────────────────────────────────────────────
    if use_gpu:
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = props["name"] if isinstance(props["name"], str) else props["name"].decode()
            vram_gb = props["totalGlobalMem"] / (1024 ** 3)
            print(f"GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        except Exception:
            print("GPU detected (could not query properties)")

        cp.cuda.Device(0).synchronize()
        print(f"\n{'='*56}")
        print(f"  GPU run: {args.total_darts:,} darts")
        print(f"{'='*56}")

        start = time.perf_counter()
        if args.chunked:
            result_gpu = estimate_pi_gpu_chunked(args.total_darts, args.seed)
        else:
            result_gpu = estimate_pi_gpu(args.total_darts, args.seed)
        cp.cuda.Device(0).synchronize()
        gpu_time = time.perf_counter() - start

        result_gpu["elapsed_seconds"] = round(gpu_time, 4)
        print(f"  Pi ≈ {result_gpu['pi_estimate']:.12f}")
        print(f"  Error: {result_gpu['error']:.2e}")
        print(f"  Time:  {gpu_time:.4f}s")
    else:
        if not GPU_AVAILABLE:
            print("No GPU detected (CuPy not installed). Running CPU only.\n")
        gpu_time = None

    # ── CPU run (for comparison) ─────────────────────────────
    cpu_time = None
    result_cpu = None
    if not args.skip_cpu:
        print(f"\n{'='*56}")
        print(f"  CPU run: {args.total_darts:,} darts")
        print(f"{'='*56}")

        start = time.perf_counter()
        result_cpu = estimate_pi_cpu(args.total_darts, args.seed)
        cpu_time = time.perf_counter() - start

        result_cpu["elapsed_seconds"] = round(cpu_time, 4)
        print(f"  Pi ≈ {result_cpu['pi_estimate']:.12f}")
        print(f"  Error: {result_cpu['error']:.2e}")
        print(f"  Time:  {cpu_time:.4f}s")
    else:
        print("\n  Skipping CPU run (--skip-cpu)")

    # ── Speedup comparison ───────────────────────────────────
    if use_gpu and gpu_time and cpu_time:
        speedup = cpu_time / gpu_time
        print(f"\n{'='*56}")
        print(f"  SPEEDUP: GPU is {speedup:.1f}x faster than CPU")
        print(f"  CPU: {cpu_time:.4f}s  vs  GPU: {gpu_time:.4f}s")
        print(f"{'='*56}")

    # ── Save results ─────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    comparison = {
        "total_darts": args.total_darts,
    }
    if result_cpu:
        comparison["cpu"] = result_cpu
    if use_gpu:
        comparison["gpu"] = result_gpu
        if cpu_time:
            comparison["speedup"] = round(cpu_time / gpu_time, 2)

    out_path = os.path.join(args.output_dir, "gpu_vs_cpu.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
