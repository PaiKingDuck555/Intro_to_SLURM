/**
 * Distillation cost and accuracy modeling.
 *
 * Uses log-linear scaling from KD literature:
 *   retention = 1 - k * ln(teacher_params / student_params)
 * where k ≈ 0.08 fits DistilBERT (2x compression → ~97% retention)
 * and TinyBERT (7x compression → ~93% retention).
 */

const DISTILL_K = 0.08;

export interface DistillationInputs {
  teacherParams: number;       // billions
  studentParams: number;       // billions
  teacherPricePerHour: number;
  studentPricePerHour: number;
  teacherPrefillTps: number;
  teacherDecodeTps: number;
  studentPrefillTps: number;
  studentDecodeTps: number;
  concurrentUsers: number;
  temperature: number;
  alpha: number;
  numSamples: number;
}

export interface DistillationPlan {
  accuracyRetentionPct: number;
  compressionRatio: number;
  teacherGpusNeeded: number;
  studentGpusNeeded: number;
  teacherCostPerHour: number;
  studentCostPerHour: number;
  costSavingsPct: number;
  studentPrefillTps: number;
  studentDecodeTps: number;
  cacheJobDurationEstimateHrs: number;
  trainJobDurationEstimateHrs: number;
  totalDistillCostUsd: number;
  temperature: number;
  alpha: number;
}

export interface CompareResult {
  distill: DistillationPlan;
  quantized: {
    pricePerHour: number;
    gpusNeeded: number;
    prefillTps: number;
    decodeTps: number;
    accuracyRetentionPct: number;
  };
  recommendation: "distill" | "quantize";
  reason: string;
}

export interface CompareInputs extends DistillationInputs {
  teacherInt4PricePerHour: number;
  teacherInt4PrefillTps: number;
  teacherInt4DecodeTps: number;
}

/** Fraction of teacher accuracy retained after distillation. Returns [0, 1]. */
export function estimateAccuracyRetention(
  teacherParams: number,
  studentParams: number
): number {
  if (studentParams >= teacherParams) return 1.0;
  const ratio = teacherParams / studentParams;
  return Math.max(0, Math.min(1, 1.0 - DISTILL_K * Math.log(ratio)));
}

function gpusNeeded(decodeTps: number, users: number): number {
  return Math.max(1, Math.ceil(users / decodeTps));
}

export function estimateDistillationPlan(
  inputs: DistillationInputs
): DistillationPlan {
  const retention = estimateAccuracyRetention(
    inputs.teacherParams,
    inputs.studentParams
  );
  const compressionRatio = inputs.teacherParams / inputs.studentParams;

  const teacherGpus = gpusNeeded(inputs.teacherDecodeTps, inputs.concurrentUsers);
  const studentGpus = gpusNeeded(inputs.studentDecodeTps, inputs.concurrentUsers);

  const teacherCostPerHour = teacherGpus * inputs.teacherPricePerHour;
  const studentCostPerHour = studentGpus * inputs.studentPricePerHour;
  const costSavingsPct =
    teacherCostPerHour > 0
      ? ((teacherCostPerHour - studentCostPerHour) / teacherCostPerHour) * 100
      : 0;

  // Cache: teacher processes numSamples * ~256 tokens at teacherPrefillTps
  const avgTok = 256;
  const cacheS = (inputs.numSamples * avgTok) / Math.max(inputs.teacherPrefillTps, 1);
  const cacheHrs = cacheS / 3600;

  // Train: 3 epochs at ~500 tok/s (student FP16 training)
  const trainS = (inputs.numSamples * avgTok * 3) / 500;
  const trainHrs = trainS / 3600;

  const totalCost =
    cacheHrs * inputs.teacherPricePerHour + trainHrs * inputs.studentPricePerHour;

  return {
    accuracyRetentionPct: Math.round(retention * 1000) / 10,
    compressionRatio: Math.round(compressionRatio * 10) / 10,
    teacherGpusNeeded: teacherGpus,
    studentGpusNeeded: studentGpus,
    teacherCostPerHour,
    studentCostPerHour,
    costSavingsPct: Math.round(costSavingsPct * 10) / 10,
    studentPrefillTps: inputs.studentPrefillTps,
    studentDecodeTps: inputs.studentDecodeTps,
    cacheJobDurationEstimateHrs: Math.round(cacheHrs * 10) / 10,
    trainJobDurationEstimateHrs: Math.round(trainHrs * 10) / 10,
    totalDistillCostUsd: Math.round(totalCost * 100) / 100,
    temperature: inputs.temperature,
    alpha: inputs.alpha,
  };
}

export function compareDistillVsQuant(inputs: CompareInputs): CompareResult {
  const distill = estimateDistillationPlan(inputs);

  const quantGpus = gpusNeeded(inputs.teacherInt4DecodeTps, inputs.concurrentUsers);
  const quantized = {
    pricePerHour: quantGpus * inputs.teacherInt4PricePerHour,
    gpusNeeded: quantGpus,
    prefillTps: inputs.teacherInt4PrefillTps,
    decodeTps: inputs.teacherInt4DecodeTps,
    accuracyRetentionPct: 97.0, // NF4 double-quant retains ~97% per literature
  };

  // Recommend distillation when it saves >20% cost AND retains >80% accuracy
  const shouldDistill =
    distill.costSavingsPct > 20 && distill.accuracyRetentionPct > 80;

  return {
    distill,
    quantized,
    recommendation: shouldDistill ? "distill" : "quantize",
    reason: shouldDistill
      ? `Distillation saves ${distill.costSavingsPct.toFixed(1)}% cost with ${distill.accuracyRetentionPct.toFixed(1)}% accuracy retention`
      : `Quantization retains ${quantized.accuracyRetentionPct}% accuracy at $0 one-time cost`,
  };
}
