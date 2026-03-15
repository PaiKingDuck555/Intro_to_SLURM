"use client";

import type { UserConstraints } from "@/lib/optimizer";
import { GPU_LIST } from "@/lib/gpu-catalog";

interface Props {
  constraints: UserConstraints;
  onChange: (c: UserConstraints) => void;
}

const GOALS = [
  { value: "cost" as const, label: "Minimize Cost", icon: "$" },
  { value: "latency" as const, label: "Minimize Latency", icon: "ms" },
  { value: "throughput" as const, label: "Max Throughput", icon: "t/s" },
  { value: "balanced" as const, label: "Balanced", icon: "=" },
];

const QUANT_OPTIONS = ["FP16", "INT8", "INT4"];

function NumberInput({
  label,
  value,
  onChange,
  suffix,
  min = 0,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  suffix?: string;
  min?: number;
}) {
  return (
    <div>
      <label className="block text-xs text-zinc-400 mb-1.5">{label}</label>
      <div className="relative">
        <input
          type="number"
          value={value}
          min={min}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                     focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
                     text-zinc-100 [appearance:textfield]"
        />
        {suffix && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-zinc-500">
            {suffix}
          </span>
        )}
      </div>
    </div>
  );
}

export function ConstraintsPanel({ constraints, onChange }: Props) {
  const update = (partial: Partial<UserConstraints>) =>
    onChange({ ...constraints, ...partial });

  const toggleQuant = (q: string) => {
    const current = constraints.allowedQuant;
    const next = current.includes(q)
      ? current.filter((x) => x !== q)
      : [...current, q];
    if (next.length > 0) update({ allowedQuant: next });
  };

  const toggleGpu = (id: string) => {
    const current = constraints.allowedGpus;
    const next = current.includes(id)
      ? current.filter((x) => x !== id)
      : [...current, id];
    if (next.length > 0) update({ allowedGpus: next });
  };

  return (
    <div className="space-y-6">
      {/* Goal */}
      <div>
        <label className="text-sm font-medium text-zinc-400 uppercase tracking-wider">
          Optimization Goal
        </label>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3">
          {GOALS.map((g) => (
            <button
              key={g.value}
              onClick={() => update({ goal: g.value })}
              className={`px-3 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                constraints.goal === g.value
                  ? "border-indigo-500 bg-indigo-500/15 text-indigo-300"
                  : "border-zinc-700 bg-zinc-800/50 text-zinc-400 hover:border-zinc-500"
              }`}
            >
              <span className="block text-xs opacity-60 mb-0.5">{g.icon}</span>
              {g.label}
            </button>
          ))}
        </div>
      </div>

      {/* Constraints */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
        <NumberInput
          label="Max Latency (first token)"
          value={constraints.maxLatencyMs}
          onChange={(v) => update({ maxLatencyMs: v })}
          suffix="ms"
          min={1}
        />
        <NumberInput
          label="Monthly Budget"
          value={constraints.maxMonthlyBudget}
          onChange={(v) => update({ maxMonthlyBudget: v })}
          suffix="$/mo"
          min={100}
        />
        <NumberInput
          label="Concurrent Users"
          value={constraints.concurrentUsers}
          onChange={(v) => update({ concurrentUsers: v })}
          min={1}
        />
        <NumberInput
          label="Avg Prompt Length"
          value={constraints.avgPromptTokens}
          onChange={(v) => update({ avgPromptTokens: v })}
          suffix="tokens"
          min={1}
        />
        <NumberInput
          label="Avg Output Length"
          value={constraints.avgOutputTokens}
          onChange={(v) => update({ avgOutputTokens: v })}
          suffix="tokens"
          min={1}
        />
      </div>

      {/* Quantization */}
      <div className="flex flex-wrap gap-6">
        <div>
          <label className="block text-xs text-zinc-400 mb-2">
            Allowed Quantization
          </label>
          <div className="flex gap-2">
            {QUANT_OPTIONS.map((q) => (
              <button
                key={q}
                onClick={() => toggleQuant(q)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-all ${
                  constraints.allowedQuant.includes(q)
                    ? q === "FP16"
                      ? "border-blue-500 bg-blue-500/15 text-blue-300"
                      : q === "INT8"
                      ? "border-emerald-500 bg-emerald-500/15 text-emerald-300"
                      : "border-amber-500 bg-amber-500/15 text-amber-300"
                    : "border-zinc-700 bg-zinc-800/50 text-zinc-500"
                }`}
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-xs text-zinc-400 mb-2">
            Allowed GPUs
          </label>
          <div className="flex flex-wrap gap-2">
            {GPU_LIST.map((g) => (
              <button
                key={g.id}
                onClick={() => toggleGpu(g.id)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-all ${
                  constraints.allowedGpus.includes(g.id)
                    ? "border-indigo-500 bg-indigo-500/15 text-indigo-300"
                    : "border-zinc-700 bg-zinc-800/50 text-zinc-500"
                }`}
              >
                {g.name.replace("NVIDIA ", "")}
                <span className="opacity-50 ml-1">${g.price_per_hour}/hr</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
