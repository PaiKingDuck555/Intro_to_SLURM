import { GPU_CATALOG, GPU_LIST, type GpuSpec } from "./gpu-catalog";
import { getBenchmarksForModel, type BenchmarkEntry } from "./benchmark-data";

export interface UserConstraints {
  model: string;
  goal: "cost" | "latency" | "pareto";
  maxLatencyMs: number;
  maxMonthlyBudget: number;
  concurrentUsers: number;
  avgPromptTokens: number;
  avgOutputTokens: number;
  allowedQuant: string[];
  allowedGpus: string[];
}

export interface GpuEstimate {
  gpuId: string;
  gpuName: string;
  pricePerHour: number;
  vramGb: number;
  fitsInVram: boolean;
  prefillTps: number;
  decodeTps: number;
  source: "measured" | "estimated";
}

export interface Architecture {
  id: string;
  label: string;
  quant: string;
  prefillGpu: GpuEstimate;
  decodeGpu: GpuEstimate;
  prefillGpuCount: number;
  decodeGpuCount: number;
  disaggregated: boolean;

  prefillLatencyMs: number;
  decodeLatencyMsPerToken: number;
  totalLatencyMs: number;
  totalThroughputTps: number;

  hourlySpend: number;
  monthlySpend: number;
  costPer1kPrefill: number;
  costPer1kDecode: number;

  meetsLatency: boolean;
  meetsBudget: boolean;
  meetsAll: boolean;

  paretoScore: number;
  reason: string;
}

/*──────────────────────────────────────────────────────────────
  ESTIMATION — scale B200 benchmark numbers to other GPUs
  
  Prefill is compute-bound  → scale by FLOPS ratio
  Decode  is bandwidth-bound → scale by memory bandwidth ratio

    prefill_tps_target = prefill_tps_b200 × (target_flops / b200_flops)
    decode_tps_target  = decode_tps_b200  × (target_bw    / b200_bw)

  VRAM fit check: model must use ≤ 85% of GPU VRAM (leave
  headroom for KV cache and framework overhead).
──────────────────────────────────────────────────────────────*/
function estimateOnGpu(
  bench: BenchmarkEntry,
  targetGpuId: string
): GpuEstimate {
  const gpu = GPU_CATALOG[targetGpuId];
  if (!gpu) throw new Error(`Unknown GPU: ${targetGpuId}`);

  const b200 = GPU_CATALOG["B200"];

  if (bench.gpu.includes("B200") && targetGpuId === "B200") {
    return {
      gpuId: targetGpuId,
      gpuName: gpu.name,
      pricePerHour: gpu.price_per_hour,
      vramGb: gpu.vram_gb,
      fitsInVram: true,
      prefillTps: bench.prefill_tps,
      decodeTps: bench.decode_tps,
      source: "measured",
    };
  }

  const flopsRatio = gpu.flops_fp16_tflops / b200.flops_fp16_tflops;
  const bwRatio = gpu.bandwidth_tb_s / b200.bandwidth_tb_s;
  const fits = bench.model_gb <= gpu.vram_gb * 0.85;

  return {
    gpuId: targetGpuId,
    gpuName: gpu.name,
    pricePerHour: gpu.price_per_hour,
    vramGb: gpu.vram_gb,
    fitsInVram: fits,
    prefillTps: Math.round(bench.prefill_tps * flopsRatio),
    decodeTps: Math.round(bench.decode_tps * bwRatio),
    source: "estimated",
  };
}

/*──────────────────────────────────────────────────────────────
  COST PER 1K TOKENS
  
    cost_per_1k = (price_per_hour / 3600 / tps) × 1000
    
  i.e. how much it costs to process 1000 tokens at the given
  throughput rate, paying the given hourly GPU price.
──────────────────────────────────────────────────────────────*/
function costPer1k(pricePerHour: number, tps: number): number {
  if (tps <= 0) return Infinity;
  return (pricePerHour / 3600 / tps) * 1000;
}

/*──────────────────────────────────────────────────────────────
  MAIN OPTIMIZER
  
  For each (quant × prefillGpu × decodeGpu) combination:

  1. PREFILL LATENCY (time to first token):
     prefill_ms = (prompt_tokens / prefill_tps) × 1000

  2. PREFILL GPU COUNT (to meet latency target):
     if prefill_ms > maxLatency → need multiple GPUs to
     shard the prompt across them in parallel:
       prefill_gpu_count = ⌈ prefill_ms / maxLatency ⌉
       effective_latency = prefill_ms / prefill_gpu_count

  3. DECODE GPU COUNT (to serve concurrent users):
     Each user generates ~1 token/step, so N concurrent users
     need N tokens/sec total decode bandwidth:
       decode_gpu_count = ⌈ concurrent_users / decode_tps ⌉

  4. TOTAL LATENCY (full request):
     total_ms = effective_prefill_ms
              + (avg_output_tokens / (decode_tps × decode_gpu_count)) × 1000

  5. HOURLY / MONTHLY COST:
     hourly  = prefill_gpus × prefill_price + decode_gpus × decode_price
     monthly = hourly × 720  (24h × 30d)

  6. CONSTRAINT CHECKS:
     meetsLatency = effective_prefill_ms ≤ maxLatencyMs
     meetsBudget  = monthlySpend ≤ maxMonthlyBudget

  7. SORTING:
     - "cost"    → sort by monthlySpend ↑
     - "latency" → sort by prefillLatencyMs ↑
     - "pareto"  → sort by normalized distance from origin:
         cost_norm    = (cost - min_cost) / (max_cost - min_cost)
         latency_norm = (lat  - min_lat)  / (max_lat  - min_lat)
         d = √(cost_norm² + latency_norm²)
       This finds the "knee" of the Pareto frontier — the
       architecture closest to ideal (0 cost, 0 latency).
──────────────────────────────────────────────────────────────*/
export function optimize(constraints: UserConstraints): Architecture[] {
  const safeUsers = Math.max(1, constraints.concurrentUsers);
  const safePrompt = Math.max(1, constraints.avgPromptTokens);
  const safeOutput = Math.max(1, constraints.avgOutputTokens);

  const architectures: Architecture[] = [];

  for (const quant of constraints.allowedQuant) {
    const quantKey =
      quant === "FP16" ? "none" : (quant.toLowerCase() as "int8" | "int4");
    const benchmarks = getBenchmarksForModel(constraints.model, quantKey);
    if (benchmarks.length === 0) continue;

    const bestPrefill = benchmarks.reduce((a, b) =>
      a.prefill_tps > b.prefill_tps ? a : b
    );
    const bestDecode = benchmarks.reduce((a, b) =>
      a.decode_tps > b.decode_tps ? a : b
    );

    for (const prefillGpuId of constraints.allowedGpus) {
      for (const decodeGpuId of constraints.allowedGpus) {
        const pEst = estimateOnGpu(bestPrefill, prefillGpuId);
        const dEst = estimateOnGpu(bestDecode, decodeGpuId);

        if (!pEst.fitsInVram || !dEst.fitsInVram) continue;

        // ── Step 1: Prefill latency ──
        const rawPrefillMs = (safePrompt / pEst.prefillTps) * 1000;

        // ── Step 2: Prefill GPU count ──
        let prefillGpuCount = 1;
        if (rawPrefillMs > constraints.maxLatencyMs) {
          prefillGpuCount = Math.ceil(rawPrefillMs / constraints.maxLatencyMs);
        }
        const effectivePrefillMs = rawPrefillMs / prefillGpuCount;

        // ── Step 3: Decode GPU count ──
        const decodeGpuCount = Math.max(
          1,
          Math.ceil(safeUsers / dEst.decodeTps)
        );

        // ── Step 4: Total latency ──
        const totalLatencyMs =
          effectivePrefillMs +
          (safeOutput / (dEst.decodeTps * decodeGpuCount)) * 1000;

        // ── Step 5: Cost ──
        const disaggregated = prefillGpuId !== decodeGpuId || quant !== "FP16";
        const hourlySpend =
          prefillGpuCount * pEst.pricePerHour +
          decodeGpuCount * dEst.pricePerHour;
        const monthlySpend = hourlySpend * 720;

        // ── Step 6: Constraint checks ──
        const meetsLatency = effectivePrefillMs <= constraints.maxLatencyMs;
        const meetsBudget = monthlySpend <= constraints.maxMonthlyBudget;

        let reason = "";
        if (disaggregated) {
          reason = `${quant} quantization`;
          if (prefillGpuId !== decodeGpuId) {
            reason += `, different GPUs for prefill/decode`;
          }
        } else {
          reason = "Unified architecture, no disaggregation";
        }

        architectures.push({
          id: `${quant}-${prefillGpuId}-${decodeGpuId}`,
          label: disaggregated
            ? `${pEst.gpuName} (P) + ${dEst.gpuName} (D) [${quant}]`
            : `${pEst.gpuName} [${quant}]`,
          quant,
          prefillGpu: pEst,
          decodeGpu: dEst,
          prefillGpuCount,
          decodeGpuCount,
          disaggregated,
          prefillLatencyMs: effectivePrefillMs,
          decodeLatencyMsPerToken:
            dEst.decodeTps > 0 ? (1 / dEst.decodeTps) * 1000 : Infinity,
          totalLatencyMs,
          totalThroughputTps: dEst.decodeTps * decodeGpuCount,
          hourlySpend,
          monthlySpend,
          costPer1kPrefill: costPer1k(pEst.pricePerHour, pEst.prefillTps),
          costPer1kDecode: costPer1k(dEst.pricePerHour, dEst.decodeTps),
          meetsLatency,
          meetsBudget,
          meetsAll: meetsLatency && meetsBudget,
          paretoScore: 0,
          reason,
        });
      }
    }
  }

  // ── Step 7: Compute Pareto scores ──
  if (architectures.length > 0) {
    const minCost = Math.min(...architectures.map((a) => a.monthlySpend));
    const maxCost = Math.max(...architectures.map((a) => a.monthlySpend));
    const minLat = Math.min(...architectures.map((a) => a.prefillLatencyMs));
    const maxLat = Math.max(...architectures.map((a) => a.prefillLatencyMs));

    const costRange = maxCost - minCost || 1;
    const latRange = maxLat - minLat || 1;

    for (const arch of architectures) {
      const cNorm = (arch.monthlySpend - minCost) / costRange;
      const lNorm = (arch.prefillLatencyMs - minLat) / latRange;
      arch.paretoScore = Math.sqrt(cNorm * cNorm + lNorm * lNorm);
    }
  }

  // ── Sorting ──
  const sortFn = {
    cost: (a: Architecture, b: Architecture) => {
      if (a.meetsAll !== b.meetsAll) return a.meetsAll ? -1 : 1;
      return a.monthlySpend - b.monthlySpend;
    },
    latency: (a: Architecture, b: Architecture) => {
      if (a.meetsAll !== b.meetsAll) return a.meetsAll ? -1 : 1;
      return a.prefillLatencyMs - b.prefillLatencyMs;
    },
    pareto: (a: Architecture, b: Architecture) => {
      if (a.meetsAll !== b.meetsAll) return a.meetsAll ? -1 : 1;
      return a.paretoScore - b.paretoScore;
    },
  };

  architectures.sort(sortFn[constraints.goal]);

  return architectures;
}

export interface FineTuneEstimate {
  model: string;
  datasetTokens: number;
  epochs: number;
  method: "full" | "lora" | "qlora";
  contextLength: number;

  gpuRecommendation: string;
  gpuCount: number;
  vramRequired: number;
  estimatedHours: number;
  estimatedCost: number;
  checkpointSizeGb: number;
}

export function estimateFineTune(
  modelId: string,
  params_b: number,
  modelSizeGb: number,
  datasetTokens: number,
  epochs: number,
  method: "full" | "lora" | "qlora",
  contextLength: number,
  maxBudget: number
): FineTuneEstimate {
  const vramMultiplier = { full: 4.0, lora: 1.5, qlora: 0.6 };
  const vramRequired = modelSizeGb * vramMultiplier[method];

  const candidates = GPU_LIST.filter(
    (g) => g.vram_gb * 0.9 >= vramRequired
  ).sort((a, b) => a.price_per_hour - b.price_per_hour);

  let gpu: GpuSpec;
  let gpuCount: number;

  if (candidates.length > 0) {
    gpu = candidates[0];
    gpuCount = 1;
  } else {
    gpu =
      GPU_LIST.filter((g) => g.vram_gb >= 80).sort(
        (a, b) => a.price_per_hour - b.price_per_hour
      )[0] || GPU_LIST[0];
    gpuCount = Math.ceil(vramRequired / (gpu.vram_gb * 0.85));
  }

  const baseSpeed = { full: 300, lora: 1000, qlora: 800 };
  const gpuSpeedFactor =
    gpu.flops_fp16_tflops / GPU_CATALOG["B200"].flops_fp16_tflops;
  const tokensPerSec =
    baseSpeed[method] * gpuSpeedFactor * gpuCount * (1 / (params_b / 10));

  const totalTokens = datasetTokens * epochs;
  const estimatedSeconds = totalTokens / tokensPerSec;
  const estimatedHours = estimatedSeconds / 3600;
  const estimatedCost = estimatedHours * gpu.price_per_hour * gpuCount;

  const checkpointMultiplier = { full: 1.0, lora: 0.05, qlora: 0.03 };
  const checkpointSizeGb = modelSizeGb * checkpointMultiplier[method];

  return {
    model: modelId,
    datasetTokens,
    epochs,
    method,
    contextLength,
    gpuRecommendation: `${gpuCount}x ${gpu.name}`,
    gpuCount,
    vramRequired: Math.round(vramRequired * 10) / 10,
    estimatedHours: Math.round(estimatedHours * 10) / 10,
    estimatedCost: Math.round(estimatedCost),
    checkpointSizeGb: Math.round(checkpointSizeGb * 10) / 10,
  };
}
