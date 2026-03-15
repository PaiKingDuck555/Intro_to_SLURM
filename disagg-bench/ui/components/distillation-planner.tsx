"use client";

import { useState, useMemo } from "react";
import { compareDistillVsQuant, type CompareResult } from "@/lib/distillation";
import { getBenchmarksForModel } from "@/lib/benchmark-data";

const PAIRS = [
  {
    teacher: "facebook/opt-66b",
    student: "facebook/opt-1.3b",
    label: "OPT-66B → OPT-1.3B (51× compression)",
  },
  {
    teacher: "facebook/opt-66b",
    student: "facebook/opt-30b",
    label: "OPT-66B → OPT-30B (2.2× compression)",
  },
  {
    teacher: "facebook/opt-30b",
    student: "facebook/opt-1.3b",
    label: "OPT-30B → OPT-1.3B (23× compression)",
  },
];

const GPU_PRICE = 45.0; // B200 on FluidStack, $/hr

function MetricCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: "green" | "amber" | "red" | "blue";
}) {
  const cls = {
    green: "text-emerald-400",
    amber: "text-amber-400",
    red: "text-red-400",
    blue: "text-indigo-400",
  };
  return (
    <div className="bg-zinc-800/80 rounded-lg p-3">
      <div className="text-xs text-zinc-500">{label}</div>
      <div className={`text-lg font-semibold mt-1 ${color ? cls[color] : ""}`}>
        {value}
      </div>
      {sub && <div className="text-xs text-zinc-500 mt-0.5">{sub}</div>}
    </div>
  );
}

export function DistillationPlanner() {
  const [pairIdx, setPairIdx] = useState(0);
  const [temperature, setTemperature] = useState(4.0);
  const [alpha, setAlpha] = useState(0.5);
  const [numSamples, setNumSamples] = useState(10000);
  const [concurrentUsers, setConcurrentUsers] = useState(100);

  const result: CompareResult | null = useMemo(() => {
    const pair = PAIRS[pairIdx];
    const tBenches = getBenchmarksForModel(pair.teacher, "none");
    const sBenches = getBenchmarksForModel(pair.student, "none");
    const tInt4 = getBenchmarksForModel(pair.teacher, "int4");

    if (!tBenches.length || !sBenches.length) return null;

    const bestT = tBenches.reduce((a, b) => (a.decode_tps > b.decode_tps ? a : b));
    const bestS = sBenches.reduce((a, b) => (a.decode_tps > b.decode_tps ? a : b));
    // Fallback to FP16 teacher data if no INT4 data exists for this teacher
    const bestTInt4 = tInt4.length
      ? tInt4.reduce((a, b) => (a.decode_tps > b.decode_tps ? a : b))
      : bestT;

    return compareDistillVsQuant({
      teacherParams: bestT.params_b,
      studentParams: bestS.params_b,
      teacherPricePerHour: GPU_PRICE,
      studentPricePerHour: GPU_PRICE,
      teacherInt4PricePerHour: GPU_PRICE,
      teacherPrefillTps: bestT.prefill_tps,
      teacherDecodeTps: bestT.decode_tps,
      teacherInt4PrefillTps: bestTInt4.prefill_tps,
      teacherInt4DecodeTps: bestTInt4.decode_tps,
      studentPrefillTps: bestS.prefill_tps,
      studentDecodeTps: bestS.decode_tps,
      concurrentUsers,
      temperature,
      alpha,
      numSamples,
    });
  }, [pairIdx, temperature, alpha, numSamples, concurrentUsers]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold">Distillation Planner</h2>
        <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-900/50 text-indigo-300 border border-indigo-700/50">
          NEW
        </span>
        <span className="text-xs text-zinc-500 hidden sm:inline">
          Model distillation vs. quantization — cost &amp; accuracy trade-offs
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Inputs ── */}
        <div className="space-y-5 bg-zinc-800/30 border border-zinc-700/50 rounded-xl p-5">
          <div>
            <label className="block text-xs text-zinc-400 mb-1.5">
              Teacher → Student Pair
            </label>
            <select
              value={pairIdx}
              onChange={(e) => setPairIdx(Number(e.target.value))}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
            >
              {PAIRS.map((p, i) => (
                <option key={i} value={i}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Temperature (τ)
              </label>
              <input
                type="number"
                min={1}
                max={20}
                step={0.5}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
              />
              <div className="text-xs text-zinc-600 mt-1">Sweet spot: 2–8</div>
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Alpha (α) — CE weight
              </label>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
              />
              <div className="text-xs text-zinc-600 mt-1">0 = pure KL, 1 = pure CE</div>
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Dataset Samples
              </label>
              <input
                type="number"
                value={numSamples}
                step={1000}
                onChange={(e) => setNumSamples(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Concurrent Users
              </label>
              <input
                type="number"
                value={concurrentUsers}
                onChange={(e) => setConcurrentUsers(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
              />
            </div>
          </div>

          <div className="bg-zinc-900/50 rounded-lg p-3 border border-zinc-700/50 text-xs text-zinc-400 space-y-1">
            <div className="font-medium text-zinc-300 mb-1">SLURM Pipeline</div>
            <div>
              1.{" "}
              <code className="text-indigo-400">submit_cache_logits.slurm</code>{" "}
              — teacher caches soft labels (10 parallel shards)
            </div>
            <div>
              2.{" "}
              <code className="text-indigo-400">submit_distill_student.slurm</code>{" "}
              — student trains from cache (no teacher in VRAM)
            </div>
          </div>
        </div>

        {/* ── Results ── */}
        <div className="bg-zinc-800/30 border border-zinc-700/50 rounded-xl p-5">
          {result ? (
            <div className="space-y-4">
              {/* Recommendation banner */}
              <div
                className={`rounded-lg p-3 border text-sm font-medium ${
                  result.recommendation === "distill"
                    ? "bg-emerald-900/20 border-emerald-700/50 text-emerald-300"
                    : "bg-amber-900/20 border-amber-700/50 text-amber-300"
                }`}
              >
                {result.recommendation === "distill"
                  ? "✓ Recommend: Distill"
                  : "✓ Recommend: Quantize"}{" "}
                — {result.reason}
              </div>

              {/* Side-by-side */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <div className="text-xs text-zinc-500 uppercase tracking-wider font-medium">
                    Distillation
                  </div>
                  <MetricCard
                    label="Accuracy Retention"
                    value={`${result.distill.accuracyRetentionPct}%`}
                    color={
                      result.distill.accuracyRetentionPct > 85
                        ? "green"
                        : result.distill.accuracyRetentionPct > 70
                        ? "amber"
                        : "red"
                    }
                  />
                  <MetricCard
                    label="Inference Cost/hr"
                    value={`$${result.distill.studentCostPerHour.toFixed(2)}`}
                    sub={`${result.distill.studentGpusNeeded}× GPU`}
                    color="blue"
                  />
                  <MetricCard
                    label="Cost Savings vs Teacher"
                    value={`${result.distill.costSavingsPct.toFixed(1)}%`}
                    color="green"
                  />
                  <MetricCard
                    label="One-time Distill Cost"
                    value={`$${result.distill.totalDistillCostUsd.toFixed(2)}`}
                    sub={`${result.distill.cacheJobDurationEstimateHrs.toFixed(1)}h cache + ${result.distill.trainJobDurationEstimateHrs.toFixed(1)}h train`}
                  />
                  <MetricCard
                    label="Compression"
                    value={`${result.distill.compressionRatio}×`}
                  />
                </div>
                <div className="space-y-2">
                  <div className="text-xs text-zinc-500 uppercase tracking-wider font-medium">
                    Teacher + INT4 Quant
                  </div>
                  <MetricCard
                    label="Accuracy Retention"
                    value={`${result.quantized.accuracyRetentionPct}%`}
                    color="green"
                  />
                  <MetricCard
                    label="Inference Cost/hr"
                    value={`$${result.quantized.pricePerHour.toFixed(2)}`}
                    sub={`${result.quantized.gpusNeeded}× GPU`}
                    color="amber"
                  />
                  <MetricCard
                    label="Cost Savings vs FP16"
                    value={`${(
                      ((result.distill.teacherCostPerHour -
                        result.quantized.pricePerHour) /
                        result.distill.teacherCostPerHour) *
                      100
                    ).toFixed(1)}%`}
                  />
                  <MetricCard
                    label="One-time Cost"
                    value="$0"
                    sub="No training required"
                    color="green"
                  />
                  <MetricCard
                    label="Decode Throughput"
                    value={`${result.quantized.decodeTps.toLocaleString()} tok/s`}
                  />
                </div>
              </div>

              <div className="text-xs text-zinc-500 bg-zinc-900/50 rounded-lg p-3 border border-zinc-700/50">
                <span className="text-amber-400 font-medium">Note:</span>{" "}
                Accuracy uses log-linear scaling (k=0.08) from KD literature.
                Run{" "}
                <code className="text-indigo-400">submit_cache_logits.slurm</code>{" "}
                +{" "}
                <code className="text-indigo-400">submit_distill_student.slurm</code>{" "}
                for real measured numbers.
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-zinc-500 text-sm">
              No benchmark data for selected pair
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
