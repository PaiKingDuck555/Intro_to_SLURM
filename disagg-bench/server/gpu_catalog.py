"""
GPU specifications and pricing database.

Pricing is configurable — users can override defaults with their own
rates from any cloud provider. Real B200 benchmark data is measured
on the cluster. Other GPU performance is estimated by scaling B200
numbers using known hardware ratios.

Cost per hour sources (defaults):
  These are approximate market rates. Override with --pricing flag
  or by editing the catalog. Prices change frequently.
  
  FluidStack (Mar 2026):  A100 ~$1.50, H100 ~$3.50
  Lambda Labs:            A100 ~$1.29, H100 ~$2.49
  AWS on-demand:          A100 ~$12,   H100 ~$30
  
  We default to mid-range cloud pricing as a reasonable baseline.
"""

import json
import os

GPU_CATALOG = {
    "B200": {
        "name": "NVIDIA B200",
        "vram_gb": 183,
        "bandwidth_tb_s": 8.0,
        "flops_fp16_tflops": 2250,
        "tdp_watts": 1000,
        "price_per_hour": 45.0,
    },
    "H100_SXM": {
        "name": "NVIDIA H100 SXM",
        "vram_gb": 80,
        "bandwidth_tb_s": 3.35,
        "flops_fp16_tflops": 990,
        "tdp_watts": 700,
        "price_per_hour": 30.0,
    },
    "H100_PCIe": {
        "name": "NVIDIA H100 PCIe",
        "vram_gb": 80,
        "bandwidth_tb_s": 2.0,
        "flops_fp16_tflops": 756,
        "tdp_watts": 350,
        "price_per_hour": 20.0,
    },
    "A100_80GB": {
        "name": "NVIDIA A100 80GB SXM",
        "vram_gb": 80,
        "bandwidth_tb_s": 2.0,
        "flops_fp16_tflops": 312,
        "tdp_watts": 400,
        "price_per_hour": 10.0,
    },
    "A100_40GB": {
        "name": "NVIDIA A100 40GB",
        "vram_gb": 40,
        "bandwidth_tb_s": 1.6,
        "flops_fp16_tflops": 312,
        "tdp_watts": 400,
        "price_per_hour": 6.0,
    },
    "L40S": {
        "name": "NVIDIA L40S",
        "vram_gb": 48,
        "bandwidth_tb_s": 0.864,
        "flops_fp16_tflops": 362,
        "tdp_watts": 350,
        "price_per_hour": 4.0,
    },
}

# Pre-compute ratios relative to B200
_B200 = GPU_CATALOG["B200"]
for gpu_id, gpu in GPU_CATALOG.items():
    gpu["bandwidth_ratio_vs_b200"] = gpu["bandwidth_tb_s"] / _B200["bandwidth_tb_s"]
    gpu["flops_ratio_vs_b200"] = gpu["flops_fp16_tflops"] / _B200["flops_fp16_tflops"]


def load_custom_pricing(path: str):
    """
    Override default pricing from a JSON file.
    Format: {"B200": 45.0, "A100_80GB": 10.0, ...}
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        prices = json.load(f)
    for gpu_id, price in prices.items():
        if gpu_id in GPU_CATALOG:
            GPU_CATALOG[gpu_id]["price_per_hour"] = price


def estimate_performance(benchmark: dict, target_gpu: str) -> dict:
    """
    Estimate a target GPU's performance by scaling real benchmark data.

    Prefill is compute-bound → scale by FLOPS ratio.
    Decode is memory-bound → scale by bandwidth ratio.
    
    For quantized models (INT8/INT4), the model_size_gb already reflects
    the compressed size, so VRAM fit checks work correctly.
    
    If the benchmark was run on the target GPU, returns actual measured values.
    """
    gpu = GPU_CATALOG[target_gpu]
    quant = benchmark.get("quantization", "none")
    
    # Detect which GPU the benchmark was run on
    bench_gpu = None
    for gid, g in GPU_CATALOG.items():
        if g["name"] in benchmark.get("gpu_name", ""):
            bench_gpu = gid
            break

    if bench_gpu == target_gpu:
        return {
            "gpu": target_gpu,
            "gpu_name": gpu["name"],
            "price_per_hour": gpu["price_per_hour"],
            "vram_gb": gpu["vram_gb"],
            "fits_in_vram": True,
            "quantization": quant,
            "estimated_prefill_tokens_per_sec": benchmark["prefill_tokens_per_sec"],
            "estimated_decode_tokens_per_sec": benchmark["decode_tokens_per_sec"],
            "source": "measured",
        }

    # Scale from benchmark GPU to target GPU
    if bench_gpu:
        bench_spec = GPU_CATALOG[bench_gpu]
        flops_ratio = gpu["flops_fp16_tflops"] / bench_spec["flops_fp16_tflops"]
        bw_ratio = gpu["bandwidth_tb_s"] / bench_spec["bandwidth_tb_s"]
    else:
        flops_ratio = gpu["flops_ratio_vs_b200"]
        bw_ratio = gpu["bandwidth_ratio_vs_b200"]

    # INT8 gets ~1.5x compute boost on tensor cores; INT4 gets ~2x
    quant_compute_bonus = {"none": 1.0, "int8": 1.5, "int4": 2.0}.get(quant, 1.0)
    # Quantized weights are smaller → higher effective bandwidth utilization
    quant_bw_bonus = {"none": 1.0, "int8": 1.3, "int4": 1.6}.get(quant, 1.0)

    prefill_tok_s = benchmark["prefill_tokens_per_sec"] * flops_ratio
    decode_tok_s = benchmark["decode_tokens_per_sec"] * bw_ratio

    model_size = benchmark.get("model_size_gb", 0)
    fits = model_size <= gpu["vram_gb"] * 0.85

    return {
        "gpu": target_gpu,
        "gpu_name": gpu["name"],
        "price_per_hour": gpu["price_per_hour"],
        "vram_gb": gpu["vram_gb"],
        "fits_in_vram": fits,
        "quantization": quant,
        "estimated_prefill_tokens_per_sec": round(prefill_tok_s, 1),
        "estimated_decode_tokens_per_sec": round(decode_tok_s, 1),
        "source": f"estimated from {benchmark.get('gpu_name', 'unknown')}",
    }


def cost_per_1k_tokens(price_per_hour: float, tokens_per_sec: float) -> float:
    """Calculate cost per 1K tokens given $/hr and throughput."""
    if tokens_per_sec <= 0:
        return float("inf")
    cost_per_sec = price_per_hour / 3600
    cost_per_token = cost_per_sec / tokens_per_sec
    return round(cost_per_token * 1000, 6)


def get_available_models(results_dir: str = "results") -> list[str]:
    """List all models that have benchmark data."""
    import glob
    files = glob.glob(f"{results_dir}/bench_*.json")
    models = set()
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        models.add(d["model_name"])
    return sorted(models)


def get_benchmarks_for_model(model: str, results_dir: str = "results") -> list[dict]:
    """Load all benchmark results for a specific model."""
    import glob
    files = glob.glob(f"{results_dir}/bench_*.json")
    benchmarks = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if d["model_name"] == model:
            benchmarks.append(d)
    return benchmarks


def print_catalog():
    """Print the GPU catalog with pricing."""
    print(f"{'GPU':<20} {'VRAM':>6} {'BW (TB/s)':>10} {'FLOPS':>8} {'$/hr':>8} {'TDP':>6}")
    print("─" * 64)
    for gid, g in GPU_CATALOG.items():
        print(f"{g['name']:<20} {g['vram_gb']:>4} GB {g['bandwidth_tb_s']:>8.1f}  "
              f"{g['flops_fp16_tflops']:>6.0f}T {g['price_per_hour']:>7.2f} {g['tdp_watts']:>5}W")


if __name__ == "__main__":
    print_catalog()
