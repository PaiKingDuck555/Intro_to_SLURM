import { describe, it, expect } from "vitest";
import {
  estimateAccuracyRetention,
  estimateDistillationPlan,
  compareDistillVsQuant,
} from "./distillation";

describe("estimateAccuracyRetention", () => {
  it("returns 1.0 when student params equal teacher", () => {
    expect(estimateAccuracyRetention(30, 30)).toBeCloseTo(1.0, 2);
  });

  it("returns less than 1.0 when student is smaller", () => {
    const r = estimateAccuracyRetention(30, 1.3);
    expect(r).toBeLessThan(1.0);
    expect(r).toBeGreaterThan(0.5);
  });

  it("larger gap → lower retention", () => {
    const small = estimateAccuracyRetention(66, 1.3);
    const large = estimateAccuracyRetention(66, 30);
    expect(large).toBeGreaterThan(small);
  });

  it("never returns below 0", () => {
    expect(estimateAccuracyRetention(1000, 0.1)).toBeGreaterThanOrEqual(0);
  });
});

describe("estimateDistillationPlan", () => {
  const base = {
    teacherParams: 30,
    studentParams: 1.3,
    teacherModelGb: 55.8,
    studentModelGb: 2.6,
    teacherPrefillTps: 6756,
    teacherDecodeTps: 60,
    studentPrefillTps: 15823,
    studentDecodeTps: 124,
    concurrentUsers: 100,
    temperature: 4.0,
    alpha: 0.5,
    numSamples: 10000,
  };

  it("accuracy retention is between 0 and 100", () => {
    const plan = estimateDistillationPlan(base);
    expect(plan.accuracyRetentionPct).toBeGreaterThan(0);
    expect(plan.accuracyRetentionPct).toBeLessThanOrEqual(100);
  });

  it("student costs less than teacher per hour", () => {
    const plan = estimateDistillationPlan(base);
    expect(plan.studentCostPerHour).toBeLessThan(plan.teacherCostPerHour);
  });

  it("costSavingsPct is positive", () => {
    const plan = estimateDistillationPlan(base);
    expect(plan.costSavingsPct).toBeGreaterThan(0);
  });

  it("assigns cheaper GPU to smaller student model", () => {
    const plan = estimateDistillationPlan(base);
    expect(plan.studentGpuPrice).toBeLessThan(plan.teacherGpuPrice);
  });

  it("pipeline durations are positive", () => {
    const plan = estimateDistillationPlan(base);
    expect(plan.cacheTimeMinutes).toBeGreaterThan(0);
    expect(plan.trainTimeMinutes).toBeGreaterThan(0);
  });
});

describe("compareDistillVsQuant", () => {
  const base = {
    teacherParams: 30,
    studentParams: 1.3,
    teacherModelGb: 55.8,
    studentModelGb: 2.6,
    teacherInt4ModelGb: 16.5,
    teacherPrefillTps: 6756,
    teacherDecodeTps: 60,
    teacherInt4PrefillTps: 8065,
    teacherInt4DecodeTps: 24,
    studentPrefillTps: 15823,
    studentDecodeTps: 124,
    concurrentUsers: 100,
    temperature: 4.0,
    alpha: 0.5,
    numSamples: 10000,
  };

  it("returns distill and quantized options", () => {
    const r = compareDistillVsQuant(base);
    expect(r.distill).toBeDefined();
    expect(r.quantized).toBeDefined();
  });

  it("recommendation is either distill or quantize", () => {
    const r = compareDistillVsQuant(base);
    expect(["distill", "quantize"]).toContain(r.recommendation);
  });

  it("reason string is non-empty", () => {
    const r = compareDistillVsQuant(base);
    expect(r.reason.length).toBeGreaterThan(0);
  });
});
