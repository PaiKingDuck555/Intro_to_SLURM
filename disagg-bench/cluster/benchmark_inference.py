#!/usr/bin/env python3
"""
Benchmark LLM inference: measures prefill and decode separately.

Runs real model inference on the GPU, timing each phase independently.
Supports FP16, INT8, and INT4 quantization via bitsandbytes.
Results are saved as JSON for downstream analysis by the optimizer.

Usage:
  python benchmark_inference.py \
      --model facebook/opt-1.3b \
      --batch-size 32 \
      --seq-length 512 \
      --decode-tokens 200 \
      --quantization int8 \
      --output-dir results
"""

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, asdict

import torch


@dataclass
class BenchmarkResult:
    model_name: str
    gpu_name: str
    gpu_vram_gb: float
    batch_size: int
    seq_length: int
    decode_tokens: int
    dtype: str
    quantization: str  # "none", "int8", or "int4"

    prefill_time_ms: float
    prefill_tokens_per_sec: float
    prefill_avg_power_watts: float
    prefill_max_power_watts: float
    prefill_tokens_per_watt: float

    decode_time_per_token_ms: float
    decode_tokens_per_sec: float
    decode_avg_power_watts: float
    decode_max_power_watts: float
    decode_tokens_per_watt: float

    vram_used_prefill_gb: float
    vram_used_decode_gb: float
    vram_total_gb: float

    total_params_billions: float
    model_size_gb: float  # actual size in VRAM after quantization


def get_gpu_info() -> tuple[str, float]:
    """Return (gpu_name, vram_gb)."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return name, vram


def get_gpu_power() -> float:
    """Read current GPU power draw in watts via nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return float(result.stdout.strip().split("\n")[0])
    except Exception:
        return -1.0


def sample_power_during(func, interval=0.1):
    """Run func() while sampling GPU power in a background thread. Returns (func_result, avg_watts, max_watts)."""
    import threading
    power_samples = []
    stop_event = threading.Event()

    def sampler():
        while not stop_event.is_set():
            w = get_gpu_power()
            if w > 0:
                power_samples.append(w)
            stop_event.wait(interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    result = func()
    stop_event.set()
    t.join(timeout=2)

    avg_w = sum(power_samples) / len(power_samples) if power_samples else -1
    max_w = max(power_samples) if power_samples else -1
    return result, round(avg_w, 1), round(max_w, 1)


def reset_gpu():
    """Clear GPU caches between benchmark phases."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_model(model_name: str, dtype: torch.dtype, quantization: str = "none"):
    """Load model and tokenizer onto GPU, with optional INT8/INT4 quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    quant_label = f"{dtype}, quant={quantization}"
    print(f"Loading {model_name} ({quant_label})...")
    start = time.perf_counter()

    AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "cuda:0",
    }

    if quantization == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        load_kwargs["torch_dtype"] = dtype
    elif quantization == "int4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    elapsed = time.perf_counter() - start
    num_params = sum(p.numel() for p in model.parameters())

    # For quantized models, nbytes reflects actual storage, not original FP16 size
    model_size_gb = sum(p.nbytes for p in model.parameters()) / (1024 ** 3)

    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Parameters: {num_params / 1e9:.2f}B")
    print(f"  Model size in VRAM: {model_size_gb:.1f} GB")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print(f"  Quantization: {quantization}")

    return model, tokenizer, num_params, model_size_gb


def benchmark_prefill(model, batch_size: int, seq_length: int,
                      vocab_size: int, warmup: int = 3, repeats: int = 10) -> dict:
    """
    Measure prefill: process a full prompt in one forward pass.
    Returns timing and utilization stats.
    """
    reset_gpu()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device="cuda")
    attention_mask = torch.ones_like(input_ids)

    # Warmup runs — GPU needs to JIT compile kernels, warm caches
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()

    vram_before = torch.cuda.memory_allocated()

    # Timed runs with power sampling
    def timed_prefill():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            with torch.no_grad():
                out = model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
        return time.perf_counter() - start, out

    (elapsed, outputs), avg_watts, max_watts = sample_power_during(
        lambda: timed_prefill()
    )

    vram_after = torch.cuda.max_memory_allocated()

    avg_time = elapsed / repeats
    total_tokens = batch_size * seq_length
    tokens_per_sec = total_tokens / avg_time

    del outputs, input_ids, attention_mask
    reset_gpu()

    return {
        "time_ms": round(avg_time * 1000, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "vram_used_gb": round(vram_after / (1024 ** 3), 2),
        "avg_power_watts": avg_watts,
        "max_power_watts": max_watts,
        "tokens_per_watt": round(tokens_per_sec / avg_watts, 2) if avg_watts > 0 else -1,
    }


def benchmark_decode(model, batch_size: int, seq_length: int,
                     decode_tokens: int, vocab_size: int,
                     warmup: int = 2, repeats: int = 3) -> dict:
    """
    Measure decode: generate tokens one at a time using KV cache.
    Returns timing and utilization stats.
    """
    reset_gpu()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device="cuda")
    attention_mask = torch.ones_like(input_ids)

    # Prefill to build KV cache
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    del outputs
    torch.cuda.synchronize()

    def run_decode(num_tokens, kv_cache):
        """Generate num_tokens using KV cache, return elapsed time."""
        token = next_token.clone()
        cache = kv_cache

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(num_tokens):
            with torch.no_grad():
                out = model(
                    token,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        torch.cuda.synchronize()
        return time.perf_counter() - start

    # Warmup
    for _ in range(warmup):
        # Re-prefill for clean cache each warmup
        with torch.no_grad():
            out = model(input_ids, attention_mask=attention_mask, use_cache=True)
        run_decode(5, out.past_key_values)

    vram_peak = 0

    def timed_decode_runs():
        nonlocal vram_peak
        total = 0
        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                out = model(input_ids, attention_mask=attention_mask, use_cache=True)
            elapsed = run_decode(decode_tokens, out.past_key_values)
            total += elapsed
            vram_peak = max(vram_peak, torch.cuda.max_memory_allocated())
        return total

    total_time, avg_watts, max_watts = sample_power_during(timed_decode_runs)

    avg_total = total_time / repeats
    time_per_token = avg_total / decode_tokens
    tokens_per_sec = batch_size / time_per_token

    del input_ids, attention_mask, past_kv, next_token
    reset_gpu()

    return {
        "time_per_token_ms": round(time_per_token * 1000, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "total_time_ms": round(avg_total * 1000, 1),
        "vram_used_gb": round(vram_peak / (1024 ** 3), 2),
        "avg_power_watts": avg_watts,
        "max_power_watts": max_watts,
        "tokens_per_watt": round(tokens_per_sec / avg_watts, 2) if avg_watts > 0 else -1,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM prefill & decode")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b",
                        help="HuggingFace model name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=100,
                        help="Number of tokens to generate during decode benchmark")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--quantization", type=str, default="none",
                        choices=["none", "int8", "int4"],
                        help="Quantization: none (FP16), int8, or int4 via bitsandbytes")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--prefill-repeats", type=int, default=10)
    parser.add_argument("--decode-repeats", type=int, default=3)
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    gpu_name, gpu_vram = get_gpu_info()
    print(f"GPU: {gpu_name} ({gpu_vram:.1f} GB VRAM)")
    print(f"Config: batch={args.batch_size}, seq={args.seq_length}, "
          f"decode_tokens={args.decode_tokens}, dtype={args.dtype}, "
          f"quant={args.quantization}")
    print("=" * 60)

    model, tokenizer, num_params, model_size_gb = load_model(
        args.model, dtype, args.quantization
    )
    vocab_size = model.config.vocab_size

    # ── Prefill benchmark ──────────────────────────────────────
    print(f"\nBenchmarking PREFILL: {args.batch_size} × {args.seq_length} tokens...")
    try:
        prefill = benchmark_prefill(
            model, args.batch_size, args.seq_length, vocab_size,
            repeats=args.prefill_repeats,
        )
        print(f"  Time: {prefill['time_ms']:.1f}ms")
        print(f"  Throughput: {prefill['tokens_per_sec']:,.0f} tokens/sec")
        print(f"  Power: {prefill['avg_power_watts']:.0f}W avg, {prefill['max_power_watts']:.0f}W peak")
        print(f"  Efficiency: {prefill['tokens_per_watt']:.1f} tokens/watt")
        print(f"  VRAM: {prefill['vram_used_gb']:.1f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM at batch={args.batch_size}, seq={args.seq_length}")
        prefill = {"time_ms": -1, "tokens_per_sec": 0, "vram_used_gb": -1,
                   "avg_power_watts": -1, "max_power_watts": -1, "tokens_per_watt": -1, "oom": True}
        reset_gpu()

    # ── Decode benchmark ───────────────────────────────────────
    print(f"\nBenchmarking DECODE: {args.batch_size} requests × "
          f"{args.decode_tokens} tokens each...")
    try:
        decode = benchmark_decode(
            model, args.batch_size, args.seq_length,
            args.decode_tokens, vocab_size,
            repeats=args.decode_repeats,
        )
        print(f"  Time per token: {decode['time_per_token_ms']:.2f}ms")
        print(f"  Throughput: {decode['tokens_per_sec']:,.0f} tokens/sec")
        print(f"  Power: {decode['avg_power_watts']:.0f}W avg, {decode['max_power_watts']:.0f}W peak")
        print(f"  Efficiency: {decode['tokens_per_watt']:.2f} tokens/watt")
        print(f"  Total decode time: {decode['total_time_ms']:.0f}ms")
        print(f"  VRAM: {decode['vram_used_gb']:.1f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM during decode at batch={args.batch_size}")
        decode = {"time_per_token_ms": -1, "tokens_per_sec": 0,
                  "total_time_ms": -1, "vram_used_gb": -1,
                  "avg_power_watts": -1, "max_power_watts": -1, "tokens_per_watt": -1, "oom": True}
        reset_gpu()

    # ── Build result ───────────────────────────────────────────
    result = BenchmarkResult(
        model_name=args.model,
        gpu_name=gpu_name,
        gpu_vram_gb=round(gpu_vram, 1),
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        decode_tokens=args.decode_tokens,
        dtype=args.dtype,
        quantization=args.quantization,
        prefill_time_ms=prefill["time_ms"],
        prefill_tokens_per_sec=prefill["tokens_per_sec"],
        prefill_avg_power_watts=prefill["avg_power_watts"],
        prefill_max_power_watts=prefill["max_power_watts"],
        prefill_tokens_per_watt=prefill["tokens_per_watt"],
        decode_time_per_token_ms=decode["time_per_token_ms"],
        decode_tokens_per_sec=decode["tokens_per_sec"],
        decode_avg_power_watts=decode["avg_power_watts"],
        decode_max_power_watts=decode["max_power_watts"],
        decode_tokens_per_watt=decode["tokens_per_watt"],
        vram_used_prefill_gb=prefill["vram_used_gb"],
        vram_used_decode_gb=decode["vram_used_gb"],
        vram_total_gb=round(gpu_vram, 1),
        total_params_billions=round(num_params / 1e9, 2),
        model_size_gb=round(model_size_gb, 2),
    )

    # ── Save results ──────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.split("/")[-1]
    quant_tag = f"_q{args.quantization}" if args.quantization != "none" else ""
    filename = f"bench_{model_short}_b{args.batch_size}_s{args.seq_length}{quant_tag}.json"
    out_path = os.path.join(args.output_dir, filename)

    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"  Prefill: {prefill['tokens_per_sec']:>10,.0f} tokens/sec  "
          f"({prefill['time_ms']:.1f}ms)")
    print(f"  Decode:  {decode['tokens_per_sec']:>10,.0f} tokens/sec  "
          f"({decode['time_per_token_ms']:.2f}ms/token)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
