/**
 * Distillation cost, accuracy, and ROI modeling.
 *
 * Accuracy retention uses log-linear scaling from KD literature:
 *   retention = 1 - k * ln(teacher_params / student_params)
 * where k ≈ 0.08 fits DistilBERT and TinyBERT benchmarks.
 */

const DISTILL_K = 0.08;
const B200_PRICE = 45.0;
const AVG_TOKENS_PER_SAMPLE = 256;

export interface DistillationInputs {
  teacherParams: number;
  studentParams: number;
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
  teacherMonthlyCost: number;
  studentMonthlyCost: number;
  costSavingsPct: number;
  studentPrefillTps: number;
  studentDecodeTps: number;
  cacheTimeMinutes: number;
  trainTimeMinutes: number;
  totalDistillMinutes: number;
  totalDistillCostUsd: number;
  monthlySavingsUsd: number;
  roiPaybackMonths: number;
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

/**
 * Estimate how long distillation will take (in minutes).
 * LoRA is ~3x faster than full fine-tuning for the train phase.
 */
export function estimateDistillTime(
  numSamples: number,
  teacherPrefillTps: number,
  method: "lora" | "full"
): { cacheMin: number; trainMin: number; totalMin: number } {
  const totalTokens = numSamples * AVG_TOKENS_PER_SAMPLE;

  const cacheSeconds = totalTokens / Math.max(teacherPrefillTps, 1);
  const cacheMin = cacheSeconds / 60;

  const trainTps = method === "lora" ? 1500 : 500;
  const epochs = 3;
  const trainSeconds = (totalTokens * epochs) / trainTps;
  const trainMin = trainSeconds / 60;

  return {
    cacheMin: Math.round(cacheMin * 10) / 10,
    trainMin: Math.round(trainMin * 10) / 10,
    totalMin: Math.round((cacheMin + trainMin) * 10) / 10,
  };
}

/**
 * Estimate Tinker API cost for distillation.
 * Tinker charges ~$0.09 per forward_backward call.
 * Each sample needs 1 call per epoch.
 */
export function estimateTinkerCost(
  numSamples: number,
  epochs: number = 3
): number {
  const totalCalls = numSamples * epochs;
  return Math.round(totalCalls * 0.09 * 100) / 100;
}

/**
 * Compute ROI payback period in months.
 * ROI = one_time_cost / monthly_savings
 */
export function computeROI(
  teacherMonthlyCost: number,
  studentMonthlyCost: number,
  oneTimeCost: number
): number {
  const monthlySavings = teacherMonthlyCost - studentMonthlyCost;
  if (monthlySavings <= 0) return Infinity;
  return oneTimeCost / monthlySavings;
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
  const teacherMonthlyCost = teacherCostPerHour * 720;
  const studentMonthlyCost = studentCostPerHour * 720;
  const costSavingsPct =
    teacherCostPerHour > 0
      ? ((teacherCostPerHour - studentCostPerHour) / teacherCostPerHour) * 100
      : 0;

  const timeEst = estimateDistillTime(
    inputs.numSamples,
    inputs.teacherPrefillTps,
    "lora"
  );

  const totalDistillCost =
    (timeEst.cacheMin / 60) * inputs.teacherPricePerHour +
    (timeEst.trainMin / 60) * inputs.studentPricePerHour;

  const monthlySavings = teacherMonthlyCost - studentMonthlyCost;
  const roiPayback = monthlySavings > 0 ? totalDistillCost / monthlySavings : Infinity;

  return {
    accuracyRetentionPct: Math.round(retention * 1000) / 10,
    compressionRatio: Math.round(compressionRatio * 10) / 10,
    teacherGpusNeeded: teacherGpus,
    studentGpusNeeded: studentGpus,
    teacherCostPerHour,
    studentCostPerHour,
    teacherMonthlyCost: Math.round(teacherMonthlyCost),
    studentMonthlyCost: Math.round(studentMonthlyCost),
    costSavingsPct: Math.round(costSavingsPct * 10) / 10,
    studentPrefillTps: inputs.studentPrefillTps,
    studentDecodeTps: inputs.studentDecodeTps,
    cacheTimeMinutes: timeEst.cacheMin,
    trainTimeMinutes: timeEst.trainMin,
    totalDistillMinutes: timeEst.totalMin,
    totalDistillCostUsd: Math.round(totalDistillCost * 100) / 100,
    monthlySavingsUsd: Math.round(monthlySavings),
    roiPaybackMonths: Math.round(roiPayback * 1000) / 1000,
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
    accuracyRetentionPct: 97.0,
  };

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
