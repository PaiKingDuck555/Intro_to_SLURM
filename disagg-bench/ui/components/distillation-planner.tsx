"use client";

import { useState, useMemo, useCallback } from "react";
import {
  estimateDistillationPlan,
  estimateDistillTime,
  estimateTinkerCost,
  computeROI,
  type DistillationPlan,
} from "@/lib/distillation";
import { getBenchmarksForModel } from "@/lib/benchmark-data";
import { MODEL_CATALOG, type CatalogModel } from "@/lib/model-catalog";

const GPU_PRICE = 45.0;

const DATASETS = [
  {
    id: "databricks/databricks-dolly-15k",
    name: "Dolly 15K",
    desc: "Business Q&A, summarization, classification by Databricks employees",
    maxSamples: 15000,
  },
  {
    id: "tatsu-lab/alpaca",
    name: "Alpaca 52K",
    desc: "Instruction-following examples for general assistant tasks",
    maxSamples: 52000,
  },
  {
    id: "custom",
    name: "Custom HuggingFace Dataset",
    desc: "Enter any HuggingFace dataset ID",
    maxSamples: 100000,
  },
];

const SAMPLE_PRESETS = [500, 1000, 5000, 10000, 15000];

type DistillMethod = "ssh" | "tinker";
type RunStatus = "idle" | "running" | "complete" | "error";

interface LiveResult {
  teacher_perplexity: number;
  student_perplexity: number;
  token_agreement: number;
  retention_pct: number;
  distill_time_min: number;
  distill_cost: number;
}

function MetricCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: "green" | "amber" | "red" | "blue" | "purple";
}) {
  const cls = {
    green: "text-emerald-400",
    amber: "text-amber-400",
    red: "text-red-400",
    blue: "text-indigo-400",
    purple: "text-purple-400",
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
  const [datasetIdx, setDatasetIdx] = useState(0);
  const [customDataset, setCustomDataset] = useState("");
  const [numSamples, setNumSamples] = useState(5000);
  const [teacherId, setTeacherId] = useState("facebook/opt-30b");
  const [studentId, setStudentId] = useState("facebook/opt-1.3b");
  const [method, setMethod] = useState<DistillMethod>("ssh");
  const [temperature, setTemperature] = useState(4.0);
  const [alpha, setAlpha] = useState(0.5);
  const [concurrentUsers, setConcurrentUsers] = useState(100);

  const [runStatus, setRunStatus] = useState<RunStatus>("idle");
  const [runLog, setRunLog] = useState<string[]>([]);
  const [liveResult, setLiveResult] = useState<LiveResult | null>(null);

  const teacher = MODEL_CATALOG.find((m) => m.id === teacherId);
  const student = MODEL_CATALOG.find((m) => m.id === studentId);

  const validStudents = MODEL_CATALOG.filter((m) => m.params_b < (teacher?.params_b || Infinity));
  const validTeachers = MODEL_CATALOG.filter((m) => m.params_b > (student?.params_b || 0));

  const datasetId = DATASETS[datasetIdx].id === "custom"
    ? customDataset
    : DATASETS[datasetIdx].id;

  const estimate = useMemo((): DistillationPlan | null => {
    if (!teacher || !student) return null;

    const tBenches = getBenchmarksForModel(teacherId, "none");
    const sBenches = getBenchmarksForModel(studentId, "none");

    const bestT = tBenches.length
      ? tBenches.reduce((a, b) => (a.decode_tps > b.decode_tps ? a : b))
      : null;
    const bestS = sBenches.length
      ? sBenches.reduce((a, b) => (a.decode_tps > b.decode_tps ? a : b))
      : null;

    const teacherPrefillTps = bestT?.prefill_tps || 5000;
    const teacherDecodeTps = bestT?.decode_tps || 100;
    const studentPrefillTps = bestS?.prefill_tps || 15000;
    const studentDecodeTps = bestS?.decode_tps || 500;

    return estimateDistillationPlan({
      teacherParams: teacher.params_b,
      studentParams: student.params_b,
      teacherPricePerHour: GPU_PRICE,
      studentPricePerHour: GPU_PRICE,
      teacherPrefillTps,
      teacherDecodeTps,
      studentPrefillTps,
      studentDecodeTps,
      concurrentUsers,
      temperature,
      alpha,
      numSamples,
    });
  }, [teacherId, studentId, numSamples, concurrentUsers, temperature, alpha, teacher, student]);

  const timeEst = useMemo(() => {
    const tBenches = getBenchmarksForModel(teacherId, "none");
    const prefillTps = tBenches.length
      ? tBenches.reduce((a, b) => (a.prefill_tps > b.prefill_tps ? a : b)).prefill_tps
      : 5000;
    return estimateDistillTime(numSamples, prefillTps, "lora");
  }, [teacherId, numSamples]);

  const tinkerCost = useMemo(() => estimateTinkerCost(numSamples, 3), [numSamples]);

  const addLog = useCallback((msg: string) => {
    setRunLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  }, []);

  const startDistillation = async () => {
    setRunStatus("running");
    setRunLog([]);
    setLiveResult(null);

    addLog(`Starting distillation: ${teacher?.name} → ${student?.name}`);
    addLog(`Dataset: ${datasetId} (${numSamples} samples)`);
    addLog(`Method: ${method === "ssh" ? "SSH to B200 cluster" : "Tinker API"}`);
    addLog(`Config: T=${temperature}, α=${alpha}, LoRA, 3 epochs`);
    addLog("");

    try {
      const res = await fetch("/api/distill", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          teacher: teacherId,
          student: studentId,
          dataset: datasetId,
          numSamples,
          temperature,
          alpha,
          method,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      if (data.status === "complete") {
        setLiveResult(data.result);
        setRunStatus("complete");
        addLog("Distillation complete!");
        addLog(`Teacher perplexity: ${data.result.teacher_perplexity}`);
        addLog(`Student perplexity: ${data.result.student_perplexity}`);
        addLog(`Token agreement: ${data.result.retention_pct}%`);
      } else if (data.status === "started") {
        addLog(`Job started: ${data.jobId || "running"}`);
        addLog("This may take 15-30 minutes...");
        pollForResults(data.jobId);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setRunStatus("error");
      addLog(`Error: ${msg}`);

      if (msg.includes("SSH key") || msg.includes("ssh")) {
        addLog("");
        addLog("To run distillation, set BENCHMARK_SSH_KEY in Vercel env vars.");
        addLog("Or run manually on the cluster:");
        addLog(`  python3 cache_teacher_logits.py --teacher ${teacherId} \\`);
        addLog(`    --student ${studentId} --num-samples ${numSamples} \\`);
        addLog(`    --output-dir ~/distill_cache`);
        addLog(`  python3 train_student_distill.py --student ${studentId} \\`);
        addLog(`    --cache-dir ~/distill_cache --output-dir ~/distill_out --lora`);
      }
    }
  };

  const pollForResults = async (jobId: string) => {
    let attempts = 0;
    const maxAttempts = 120;

    const poll = async () => {
      attempts++;
      try {
        const res = await fetch(`/api/distill?jobId=${jobId}`);
        const data = await res.json();

        if (data.status === "complete") {
          setLiveResult(data.result);
          setRunStatus("complete");
          addLog("Distillation complete!");
          return;
        } else if (data.status === "failed") {
          throw new Error("Distillation job failed");
        } else {
          if (attempts % 6 === 0) addLog(`Still running... (${attempts * 10}s)`);
          if (attempts < maxAttempts) {
            setTimeout(poll, 10000);
          } else {
            throw new Error("Timeout waiting for results");
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Poll error";
        setRunStatus("error");
        addLog(`Error: ${msg}`);
      }
    };

    setTimeout(poll, 10000);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold">Distillation Lab</h2>
        <span className="text-xs px-2 py-0.5 rounded-full bg-purple-900/50 text-purple-300 border border-purple-700/50">
          LIVE
        </span>
        <span className="text-xs text-zinc-500 hidden sm:inline">
          Train a smaller model to mimic a larger one — real benchmarks, real savings
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* ── Column 1: Config ── */}
        <div className="space-y-4 bg-zinc-800/30 border border-zinc-700/50 rounded-xl p-5">
          <div className="text-sm font-medium text-zinc-300 uppercase tracking-wider">
            Configuration
          </div>

          {/* Dataset */}
          <div>
            <label className="block text-xs text-zinc-400 mb-1.5">Dataset</label>
            <select
              value={datasetIdx}
              onChange={(e) => setDatasetIdx(Number(e.target.value))}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
            >
              {DATASETS.map((d, i) => (
                <option key={i} value={i}>{d.name}</option>
              ))}
            </select>
            <div className="text-[11px] text-zinc-600 mt-1">{DATASETS[datasetIdx].desc}</div>
          </div>

          {DATASETS[datasetIdx].id === "custom" && (
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">HuggingFace Dataset ID</label>
              <input
                type="text"
                value={customDataset}
                onChange={(e) => setCustomDataset(e.target.value)}
                placeholder="org/dataset-name"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
              />
            </div>
          )}

          {/* Samples slider */}
          <div>
            <label className="block text-xs text-zinc-400 mb-1.5">
              Samples: <span className="text-zinc-200 font-medium">{numSamples.toLocaleString()}</span>
            </label>
            <div className="flex gap-2">
              {SAMPLE_PRESETS.map((n) => (
                <button
                  key={n}
                  onClick={() => setNumSamples(n)}
                  className={`flex-1 px-2 py-1.5 rounded text-xs font-medium border transition-all ${
                    numSamples === n
                      ? "border-indigo-500 bg-indigo-500/15 text-indigo-300"
                      : "border-zinc-700 bg-zinc-800/50 text-zinc-500 hover:border-zinc-500"
                  }`}
                >
                  {n >= 1000 ? `${n / 1000}K` : n}
                </button>
              ))}
            </div>
          </div>

          {/* Teacher / Student */}
          <div>
            <label className="block text-xs text-zinc-400 mb-1.5">Teacher (large model)</label>
            <select
              value={teacherId}
              onChange={(e) => {
                setTeacherId(e.target.value);
                const newTeacher = MODEL_CATALOG.find((m) => m.id === e.target.value);
                if (newTeacher && student && student.params_b >= newTeacher.params_b) {
                  const valid = MODEL_CATALOG.filter((m) => m.params_b < newTeacher.params_b);
                  if (valid.length) setStudentId(valid[0].id);
                }
              }}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
            >
              {validTeachers.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.params_b}B, {m.fp16_gb} GB)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs text-zinc-400 mb-1.5">Student (small model)</label>
            <select
              value={studentId}
              onChange={(e) => setStudentId(e.target.value)}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
            >
              {validStudents.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.params_b}B, {m.fp16_gb} GB)
                </option>
              ))}
            </select>
          </div>

          {/* Advanced */}
          <details className="group">
            <summary className="text-xs text-zinc-500 cursor-pointer hover:text-zinc-400">
              Advanced parameters
            </summary>
            <div className="grid grid-cols-3 gap-3 mt-3">
              <div>
                <label className="block text-[10px] text-zinc-500 mb-1">Temp (T)</label>
                <input type="number" min={1} max={20} step={0.5} value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-100" />
              </div>
              <div>
                <label className="block text-[10px] text-zinc-500 mb-1">Alpha</label>
                <input type="number" min={0} max={1} step={0.05} value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-100" />
              </div>
              <div>
                <label className="block text-[10px] text-zinc-500 mb-1">Users</label>
                <input type="number" min={1} value={concurrentUsers}
                  onChange={(e) => setConcurrentUsers(Math.max(1, Number(e.target.value)))}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-100" />
              </div>
            </div>
          </details>
        </div>

        {/* ── Column 2: Estimates + Run ── */}
        <div className="space-y-4 bg-zinc-800/30 border border-zinc-700/50 rounded-xl p-5">
          <div className="text-sm font-medium text-zinc-300 uppercase tracking-wider">
            Run Distillation
          </div>

          {/* Method toggle */}
          <div className="flex gap-2">
            <button
              onClick={() => setMethod("ssh")}
              className={`flex-1 px-3 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                method === "ssh"
                  ? "border-indigo-500 bg-indigo-500/15 text-indigo-300"
                  : "border-zinc-700 bg-zinc-800/50 text-zinc-400 hover:border-zinc-500"
              }`}
            >
              <div className="text-xs opacity-60 mb-0.5">B200 Cluster</div>
              SSH
            </button>
            <button
              onClick={() => setMethod("tinker")}
              className={`flex-1 px-3 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                method === "tinker"
                  ? "border-purple-500 bg-purple-500/15 text-purple-300"
                  : "border-zinc-700 bg-zinc-800/50 text-zinc-400 hover:border-zinc-500"
              }`}
            >
              <div className="text-xs opacity-60 mb-0.5">Tinker API</div>
              LoRA
            </button>
          </div>

          {/* Time / cost estimate */}
          <div className="bg-zinc-900/50 rounded-lg p-3 border border-zinc-700/50 space-y-2 text-xs">
            {method === "ssh" ? (
              <>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Cache teacher logits</span>
                  <span className="text-zinc-300">~{timeEst.cacheMin} min</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Train student (LoRA)</span>
                  <span className="text-zinc-300">~{timeEst.trainMin} min</span>
                </div>
                <div className="flex justify-between border-t border-zinc-800 pt-2">
                  <span className="text-zinc-400 font-medium">Total</span>
                  <span className="text-indigo-300 font-medium">
                    ~{timeEst.totalMin} min &middot; ${((timeEst.totalMin / 60) * GPU_PRICE).toFixed(0)}
                  </span>
                </div>
              </>
            ) : (
              <>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Tinker API calls</span>
                  <span className="text-zinc-300">{(numSamples * 3).toLocaleString()}</span>
                </div>
                <div className="flex justify-between border-t border-zinc-800 pt-2">
                  <span className="text-zinc-400 font-medium">Estimated cost</span>
                  <span className="text-purple-300 font-medium">${tinkerCost.toLocaleString()}</span>
                </div>
                <div className="text-[10px] text-zinc-600">
                  $150 free credits for new accounts at tinker-docs.thinkingmachines.ai
                </div>
              </>
            )}
          </div>

          {/* Run button */}
          <button
            onClick={startDistillation}
            disabled={runStatus === "running" || !teacher || !student}
            className={`w-full py-3 rounded-lg text-sm font-semibold transition-all ${
              runStatus === "running"
                ? "bg-zinc-700 text-zinc-400 cursor-wait"
                : "bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white"
            }`}
          >
            {runStatus === "running" ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Running...
              </span>
            ) : (
              "Start Distillation"
            )}
          </button>

          {/* Log */}
          {runLog.length > 0 && (
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 max-h-40 overflow-y-auto">
              <div className="font-mono text-[11px] space-y-0.5">
                {runLog.map((line, i) => (
                  <div key={i} className={
                    line.includes("Error") ? "text-red-400"
                    : line.includes("complete") ? "text-emerald-400"
                    : line.includes("perplexity") || line.includes("agreement") ? "text-indigo-300"
                    : "text-zinc-500"
                  }>
                    {line}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Column 3: Results ── */}
        <div className="bg-zinc-800/30 border border-zinc-700/50 rounded-xl p-5">
          <div className="text-sm font-medium text-zinc-300 uppercase tracking-wider mb-4">
            {liveResult ? "Live Results" : "Estimated Results"}
          </div>

          {estimate ? (
            <div className="space-y-3">
              {/* Teacher vs Student comparison */}
              <div className="grid grid-cols-2 gap-2">
                <div className="text-center p-3 bg-zinc-900/50 rounded-lg border border-zinc-700/50">
                  <div className="text-[10px] text-zinc-500 uppercase">Teacher</div>
                  <div className="text-sm font-semibold mt-1">{teacher?.name}</div>
                  <div className="text-lg font-bold text-red-400 mt-1">
                    ${estimate.teacherMonthlyCost.toLocaleString()}
                    <span className="text-[10px] text-zinc-500">/mo</span>
                  </div>
                  {liveResult && (
                    <div className="text-xs text-zinc-400 mt-1">
                      PPL: {liveResult.teacher_perplexity}
                    </div>
                  )}
                </div>
                <div className="text-center p-3 bg-zinc-900/50 rounded-lg border border-zinc-700/50">
                  <div className="text-[10px] text-zinc-500 uppercase">Student</div>
                  <div className="text-sm font-semibold mt-1">{student?.name}</div>
                  <div className="text-lg font-bold text-emerald-400 mt-1">
                    ${estimate.studentMonthlyCost.toLocaleString()}
                    <span className="text-[10px] text-zinc-500">/mo</span>
                  </div>
                  {liveResult && (
                    <div className="text-xs text-zinc-400 mt-1">
                      PPL: {liveResult.student_perplexity}
                    </div>
                  )}
                </div>
              </div>

              {/* Metrics */}
              <MetricCard
                label="Accuracy Retention"
                value={liveResult ? `${liveResult.retention_pct}%` : `~${estimate.accuracyRetentionPct}%`}
                sub={liveResult ? "measured token agreement" : "estimated (scaling law)"}
                color={
                  (liveResult?.retention_pct || estimate.accuracyRetentionPct) > 85
                    ? "green"
                    : (liveResult?.retention_pct || estimate.accuracyRetentionPct) > 70
                    ? "amber"
                    : "red"
                }
              />

              <div className="grid grid-cols-2 gap-2">
                <MetricCard
                  label="Monthly Savings"
                  value={`$${estimate.monthlySavingsUsd.toLocaleString()}`}
                  sub={`${estimate.costSavingsPct}% reduction`}
                  color="green"
                />
                <MetricCard
                  label="One-Time Cost"
                  value={liveResult
                    ? `$${liveResult.distill_cost.toFixed(0)}`
                    : `~$${estimate.totalDistillCostUsd.toFixed(0)}`}
                  sub={`${estimate.totalDistillMinutes} min LoRA`}
                  color="blue"
                />
              </div>

              <MetricCard
                label="ROI Payback"
                value={estimate.roiPaybackMonths < 0.01
                  ? "< 1 day"
                  : estimate.roiPaybackMonths < 1
                  ? `${Math.round(estimate.roiPaybackMonths * 30)} days`
                  : `${estimate.roiPaybackMonths.toFixed(1)} months`}
                sub={`${estimate.compressionRatio}x compression`}
                color="purple"
              />

              {!liveResult && (
                <div className="text-[10px] text-zinc-600 bg-zinc-900/50 rounded-lg p-2 border border-zinc-700/50">
                  Estimates use scaling laws. Click &quot;Start Distillation&quot; for measured numbers.
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-zinc-500 text-sm">
              Select a valid teacher/student pair
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
