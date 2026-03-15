import { GPU_CATALOG, GPU_LIST, type GpuSpec } from "./gpu-catalog";
import { getBenchmarksForModel, type BenchmarkEntry } from "./benchmark-data";

export interface UserConstraints {
  model: string;
  goal: "cost" | "latency" | "throughput" | "balanced";
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

  reason: string;
}

function estimateOnGpu(
  bench: BenchmarkEntry,
  targetGpuId: string
): GpuEstimate {
  const gpu = GPU_CATALOG[targetGpuId];
  if (!gpu) throw new Error(`Unknown GPU: ${targetGpuId}`);

  const b200 = GPU_CATALOG["B200"];
  const benchIsB200 = bench.gpu.includes("B200");

  if (benchIsB200 && targetGpuId === "B200") {
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

  const baseFlops = benchIsB200
    ? b200.flops_fp16_tflops
    : b200.flops_fp16_tflops;
  const baseBw = benchIsB200 ? b200.bandwidth_tb_s : b200.bandwidth_tb_s;

  const flopsRatio = gpu.flops_fp16_tflops / baseFlops;
  const bwRatio = gpu.bandwidth_tb_s / baseBw;

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

function costPer1k(pricePerHour: number, tps: number): number {
  if (tps <= 0) return Infinity;
  return (pricePerHour / 3600 / tps) * 1000;
}

export function optimize(constraints: UserConstraints): Architecture[] {
  const architectures: Architecture[] = [];

  for (const quant of constraints.allowedQuant) {
    const quantKey =
      quant === "FP16" ? "none" : (quant.toLowerCase() as "int8" | "int4");
    const benchmarks = getBenchmarksForModel(constraints.model, quantKey);
    if (benchmarks.length === 0) continue;

    // Pick best prefill and best decode benchmarks
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

        // Prefill latency: time to process prompt
        const prefillLatencyMs =
          (constraints.avgPromptTokens / pEst.prefillTps) * 1000;

        // How many prefill GPUs to meet latency target?
        let prefillGpuCount = 1;
        if (prefillLatencyMs > constraints.maxLatencyMs) {
          prefillGpuCount = Math.ceil(
            prefillLatencyMs / constraints.maxLatencyMs
          );
        }

        // Decode: each GPU can serve dEst.decodeTps total tokens/s
        const effectivePrefillLatency = prefillLatencyMs / prefillGpuCount;

        // Decode latency per token
        const decodeLatencyMsPerToken = dEst.decodeTps > 0
          ? (1 / dEst.decodeTps) * 1000 * Math.max(1, constraints.concurrentUsers)
          : Infinity;

        // How many decode GPUs to serve concurrent users?
        const tokensPerSecNeeded = constraints.concurrentUsers;
        let decodeGpuCount = Math.max(
          1,
          Math.ceil(tokensPerSecNeeded / dEst.decodeTps)
        );

        const totalLatencyMs =
          effectivePrefillLatency +
          (constraints.avgOutputTokens / (dEst.decodeTps * decodeGpuCount)) * 1000;

        const disaggregated = prefillGpuId !== decodeGpuId || quant !== "FP16";

        const hourlySpend =
          prefillGpuCount * pEst.pricePerHour +
          decodeGpuCount * dEst.pricePerHour;
        const monthlySpend = hourlySpend * 720;

        const meetsLatency = effectivePrefillLatency <= constraints.maxLatencyMs;
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
          prefillLatencyMs: effectivePrefillLatency,
          decodeLatencyMsPerToken: dEst.decodeTps > 0 ? (1 / dEst.decodeTps) * 1000 : Infinity,
          totalLatencyMs,
          totalThroughputTps: dEst.decodeTps * decodeGpuCount,
          hourlySpend,
          monthlySpend,
          costPer1kPrefill: costPer1k(pEst.pricePerHour, pEst.prefillTps),
          costPer1kDecode: costPer1k(dEst.pricePerHour, dEst.decodeTps),
          meetsLatency,
          meetsBudget,
          meetsAll: meetsLatency && meetsBudget,
          reason,
        });
      }
    }
  }

  // Sort by optimization goal
  const sortFn = {
    cost: (a: Architecture, b: Architecture) => {
      if (a.meetsAll !== b.meetsAll) return a.meetsAll ? -1 : 1;
      return a.monthlySpend - b.monthlySpend;
    },
    latency: (a: Architecture, b: Architecture) => {
      if (a.meetsAll !== b.meetsAll) return a.meetsAll ? -1 : 1;
      return a.prefillLatencyMs - b.prefillLatencyMs;
    },
    throughput: (a: Architecture, b: Architecture) => {
      if (a.meetsAll !== b.meetsAll) return a.meetsAll ? -1 : 1;
      return b.totalThroughputTps - a.totalThroughputTps;
    },
    balanced: (a: Architecture, b: Architecture) => {
      if (a.meetsAll !== b.meetsAll) return a.meetsAll ? -1 : 1;
      const scoreA =
        a.monthlySpend / 10000 +
        a.prefillLatencyMs / 500 -
        a.totalThroughputTps / 5000;
      const scoreB =
        b.monthlySpend / 10000 +
        b.prefillLatencyMs / 500 -
        b.totalThroughputTps / 5000;
      return scoreA - scoreB;
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
  // VRAM requirement estimation
  const vramMultiplier = { full: 4.0, lora: 1.5, qlora: 0.6 };
  const vramRequired = modelSizeGb * vramMultiplier[method];

  // Find cheapest GPU that fits
  const candidates = GPU_LIST.filter(
    (g) => g.vram_gb * 0.9 >= vramRequired
  ).sort((a, b) => a.price_per_hour - b.price_per_hour);

  let gpu: GpuSpec;
  let gpuCount: number;

  if (candidates.length > 0) {
    gpu = candidates[0];
    gpuCount = 1;
  } else {
    // Need multi-GPU — use cheapest large GPU
    gpu =
      GPU_LIST.filter((g) => g.vram_gb >= 80).sort(
        (a, b) => a.price_per_hour - b.price_per_hour
      )[0] || GPU_LIST[0];
    gpuCount = Math.ceil(vramRequired / (gpu.vram_gb * 0.85));
  }

  // Training speed estimation (tokens/sec)
  // Rough: ~1000 tokens/sec/B100-equivalent for LoRA, ~300 for full finetune
  const baseSpeed = { full: 300, lora: 1000, qlora: 800 };
  const gpuSpeedFactor =
    gpu.flops_fp16_tflops / GPU_CATALOG["B200"].flops_fp16_tflops;
  const tokensPerSec =
    baseSpeed[method] * gpuSpeedFactor * gpuCount * (1 / (params_b / 10));

  const totalTokens = datasetTokens * epochs;
  const estimatedSeconds = totalTokens / tokensPerSec;
  const estimatedHours = estimatedSeconds / 3600;
  const estimatedCost = estimatedHours * gpu.price_per_hour * gpuCount;

  // Checkpoint size
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
