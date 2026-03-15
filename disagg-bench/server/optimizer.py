"""
Disaggregated inference optimizer.

Takes benchmark results + user workload requirements and recommends
the optimal GPU configuration, comparing naive (single GPU type)
vs disaggregated (different GPUs for prefill and decode).
"""

import glob
import json
import math
from dataclasses import dataclass, asdict

from gpu_catalog import GPU_CATALOG, estimate_performance


@dataclass
class WorkloadSpec:
    model: str
    concurrent_users: int
    avg_prompt_tokens: int
    avg_response_tokens: int
    max_first_token_latency_ms: float
    optimize_for: str  # "cost", "latency", or "balanced"


@dataclass
class GpuAllocation:
    gpu_type: str
    gpu_name: str
    count: int
    role: str  # "prefill", "decode", or "both"
    throughput_tokens_per_sec: float
    cost_per_hour: float
    latency_ms: float
    power_watts: float


@dataclass
class Recommendation:
    name: str
    description: str
    allocations: list
    total_cost_per_hour: float
    total_cost_per_month: float
    prefill_latency_ms: float
    decode_tokens_per_sec_per_user: float
    meets_latency_target: bool
    total_gpus: int
    total_power_watts: float
    cost_per_1k_tokens: float


def load_benchmarks(results_dir: str = "results") -> list[dict]:
    """Load all benchmark JSON files from results directory."""
    files = glob.glob(f"{results_dir}/bench_*.json")
    benchmarks = []
    for f in files:
        with open(f) as fh:
            benchmarks.append(json.load(fh))
    return benchmarks


def find_best_config(benchmarks: list[dict], target_batch: int = None) -> dict:
    """Find the benchmark config with highest decode throughput, optionally at a specific batch size."""
    valid = [b for b in benchmarks if b.get("decode_tokens_per_sec", 0) > 0]
    if target_batch:
        valid = [b for b in valid if b["batch_size"] == target_batch]
    if not valid:
        return benchmarks[0] if benchmarks else {}
    return max(valid, key=lambda b: b["decode_tokens_per_sec"])


def find_config_meeting_latency(benchmarks: list[dict],
                                 max_latency_ms: float,
                                 gpu_type: str,
                                 phase: str = "prefill") -> dict | None:
    """Find the highest-throughput config that meets a latency target."""
    valid = []
    for b in benchmarks:
        if phase == "prefill" and b.get("prefill_time_ms", -1) <= max_latency_ms:
            valid.append(b)
        elif phase == "decode" and b.get("decode_time_per_token_ms", -1) > 0:
            valid.append(b)
    if not valid:
        return None
    if phase == "prefill":
        return max(valid, key=lambda b: b["prefill_tokens_per_sec"])
    return max(valid, key=lambda b: b["decode_tokens_per_sec"])


def calculate_config(workload: WorkloadSpec, benchmarks: list[dict],
                     gpu_type: str, mode: str = "both") -> GpuAllocation | None:
    """
    Calculate how many GPUs of a given type are needed for a workload.
    mode: "both" (single GPU type), "prefill", or "decode"
    """
    gpu = GPU_CATALOG.get(gpu_type)
    if not gpu:
        return None

    best = find_best_config(benchmarks)
    if not best:
        return None

    est = estimate_performance(best, gpu_type)

    if mode in ("both", "prefill"):
        total_prefill_tokens = workload.concurrent_users * workload.avg_prompt_tokens
        prefill_tok_s = est["estimated_prefill_tokens_per_sec"]
        if prefill_tok_s <= 0:
            return None
        prefill_time_per_user = workload.avg_prompt_tokens / prefill_tok_s * 1000

        if mode == "prefill":
            gpus_for_latency = math.ceil(
                prefill_time_per_user / workload.max_first_token_latency_ms
            ) if workload.max_first_token_latency_ms > 0 else 1
            gpus_for_throughput = math.ceil(total_prefill_tokens / prefill_tok_s)
            num_gpus = max(gpus_for_latency, gpus_for_throughput, 1)

            return GpuAllocation(
                gpu_type=gpu_type,
                gpu_name=gpu["name"],
                count=num_gpus,
                role="prefill",
                throughput_tokens_per_sec=prefill_tok_s * num_gpus,
                cost_per_hour=gpu["price_per_hour"] * num_gpus,
                latency_ms=prefill_time_per_user / num_gpus,
                power_watts=gpu["tdp_watts"] * num_gpus,
            )

    if mode in ("both", "decode"):
        decode_tok_s = est["estimated_decode_tokens_per_sec"]
        if decode_tok_s <= 0:
            return None
        total_decode_demand = workload.concurrent_users  # each user needs 1 token/step

        if mode == "decode":
            num_gpus = max(math.ceil(total_decode_demand / decode_tok_s), 1)
            return GpuAllocation(
                gpu_type=gpu_type,
                gpu_name=gpu["name"],
                count=num_gpus,
                role="decode",
                throughput_tokens_per_sec=decode_tok_s * num_gpus,
                cost_per_hour=gpu["price_per_hour"] * num_gpus,
                latency_ms=best.get("decode_time_per_token_ms", 0) / est["estimated_decode_tokens_per_sec"] * best.get("decode_tokens_per_sec", 1) if est["estimated_decode_tokens_per_sec"] > 0 else 999,
                power_watts=gpu["tdp_watts"] * num_gpus,
            )

    # mode == "both": single GPU type for everything
    prefill_tok_s = est["estimated_prefill_tokens_per_sec"]
    decode_tok_s = est["estimated_decode_tokens_per_sec"]
    total_prefill = workload.concurrent_users * workload.avg_prompt_tokens
    total_decode = workload.concurrent_users

    gpus_prefill = max(math.ceil(total_prefill / prefill_tok_s), 1)
    gpus_decode = max(math.ceil(total_decode / decode_tok_s), 1)
    num_gpus = max(gpus_prefill, gpus_decode)

    prefill_latency = (workload.avg_prompt_tokens / (prefill_tok_s * num_gpus)) * 1000

    return GpuAllocation(
        gpu_type=gpu_type,
        gpu_name=gpu["name"],
        count=num_gpus,
        role="both",
        throughput_tokens_per_sec=decode_tok_s * num_gpus,
        cost_per_hour=gpu["price_per_hour"] * num_gpus,
        latency_ms=prefill_latency,
        power_watts=gpu["tdp_watts"] * num_gpus,
    )


def build_recommendation(name: str, description: str,
                          allocations: list[GpuAllocation],
                          workload: WorkloadSpec) -> Recommendation:
    """Build a complete recommendation from a list of GPU allocations."""
    total_cost = sum(a.cost_per_hour for a in allocations)
    total_gpus = sum(a.count for a in allocations)
    total_power = sum(a.power_watts for a in allocations)

    prefill_allocs = [a for a in allocations if a.role in ("prefill", "both")]
    decode_allocs = [a for a in allocations if a.role in ("decode", "both")]

    prefill_latency = min(a.latency_ms for a in prefill_allocs) if prefill_allocs else 999
    decode_throughput = sum(a.throughput_tokens_per_sec for a in decode_allocs)
    decode_per_user = decode_throughput / workload.concurrent_users if workload.concurrent_users > 0 else 0

    total_tokens_per_sec = decode_throughput
    cost_per_1k = (total_cost / 3600) / total_tokens_per_sec * 1000 if total_tokens_per_sec > 0 else float("inf")

    return Recommendation(
        name=name,
        description=description,
        allocations=[asdict(a) for a in allocations],
        total_cost_per_hour=round(total_cost, 2),
        total_cost_per_month=round(total_cost * 720, 2),
        prefill_latency_ms=round(prefill_latency, 1),
        decode_tokens_per_sec_per_user=round(decode_per_user, 1),
        meets_latency_target=prefill_latency <= workload.max_first_token_latency_ms,
        total_gpus=total_gpus,
        total_power_watts=round(total_power, 0),
        cost_per_1k_tokens=round(cost_per_1k, 6),
    )


def optimize(workload: WorkloadSpec, benchmarks: list[dict]) -> list[dict]:
    """
    Generate all candidate configurations and rank them.
    Returns a list of recommendations sorted by the optimization target.
    """
    recommendations = []
    gpu_types = list(GPU_CATALOG.keys())

    # Option 1: Naive — single GPU type for everything
    for gpu_type in gpu_types:
        alloc = calculate_config(workload, benchmarks, gpu_type, mode="both")
        if alloc:
            rec = build_recommendation(
                name=f"All {gpu_type}",
                description=f"Run both prefill and decode on {GPU_CATALOG[gpu_type]['name']}",
                allocations=[alloc],
                workload=workload,
            )
            recommendations.append(rec)

    # Option 2: Disaggregated — different GPU for prefill vs decode
    for prefill_gpu in gpu_types:
        for decode_gpu in gpu_types:
            if prefill_gpu == decode_gpu:
                continue
            p_alloc = calculate_config(workload, benchmarks, prefill_gpu, mode="prefill")
            d_alloc = calculate_config(workload, benchmarks, decode_gpu, mode="decode")
            if p_alloc and d_alloc:
                rec = build_recommendation(
                    name=f"{prefill_gpu} + {decode_gpu}",
                    description=f"Prefill on {GPU_CATALOG[prefill_gpu]['name']}, decode on {GPU_CATALOG[decode_gpu]['name']}",
                    allocations=[p_alloc, d_alloc],
                    workload=workload,
                )
                recommendations.append(rec)

    # Sort by optimization target
    if workload.optimize_for == "cost":
        recommendations.sort(key=lambda r: r.total_cost_per_hour)
    elif workload.optimize_for == "latency":
        recommendations.sort(key=lambda r: r.prefill_latency_ms)
    else:  # balanced
        recommendations.sort(key=lambda r: r.cost_per_1k_tokens)

    return [asdict(r) for r in recommendations]


def main():
    """CLI interface for the optimizer."""
    import argparse

    parser = argparse.ArgumentParser(description="Disaggregated inference optimizer")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--response-tokens", type=int, default=256)
    parser.add_argument("--max-latency-ms", type=float, default=200)
    parser.add_argument("--optimize-for", choices=["cost", "latency", "balanced"],
                        default="balanced")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    workload = WorkloadSpec(
        model=args.model,
        concurrent_users=args.users,
        avg_prompt_tokens=args.prompt_tokens,
        avg_response_tokens=args.response_tokens,
        max_first_token_latency_ms=args.max_latency_ms,
        optimize_for=args.optimize_for,
    )

    benchmarks = load_benchmarks(args.results_dir)
    if not benchmarks:
        print(f"No benchmark results found in {args.results_dir}/")
        return

    print(f"Loaded {len(benchmarks)} benchmark results")
    print(f"Workload: {workload.concurrent_users} users, "
          f"{workload.avg_prompt_tokens} prompt tokens, "
          f"{workload.avg_response_tokens} response tokens")
    print(f"Optimizing for: {workload.optimize_for}")
    print(f"Max first-token latency: {workload.max_first_token_latency_ms}ms")
    print("=" * 70)

    recs = optimize(workload, benchmarks)

    for i, rec in enumerate(recs[:5]):
        marker = " ← RECOMMENDED" if i == 0 else ""
        latency_ok = "✓" if rec["meets_latency_target"] else "✗"

        print(f"\n{'─' * 70}")
        print(f"  #{i+1}: {rec['name']}{marker}")
        print(f"  {rec['description']}")
        print(f"  GPUs: {rec['total_gpus']}")
        print(f"  Cost: ${rec['total_cost_per_hour']:.2f}/hr "
              f"(${rec['total_cost_per_month']:,.0f}/month)")
        print(f"  Prefill latency: {rec['prefill_latency_ms']:.1f}ms {latency_ok}")
        print(f"  Decode per user: {rec['decode_tokens_per_sec_per_user']:.1f} tok/s")
        print(f"  Cost per 1K tokens: ${rec['cost_per_1k_tokens']:.4f}")
        print(f"  Power: {rec['total_power_watts']:,.0f}W")

        for alloc in rec["allocations"]:
            print(f"    └─ {alloc['count']}× {alloc['gpu_name']} ({alloc['role']}): "
                  f"${alloc['cost_per_hour']:.2f}/hr")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(recs, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
