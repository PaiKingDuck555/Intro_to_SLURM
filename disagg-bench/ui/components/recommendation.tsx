"use client";

import type { Architecture } from "@/lib/optimizer";

interface Props {
  architectures: Architecture[];
}

function Badge({
  ok,
  label,
  value,
}: {
  ok: boolean;
  label: string;
  value: string;
}) {
  return (
    <div
      className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm ${
        ok ? "bg-emerald-900/30 text-emerald-300" : "bg-red-900/30 text-red-300"
      }`}
    >
      <span className="text-base">{ok ? "\u2713" : "\u2717"}</span>
      <span className="text-zinc-400">{label}:</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}

function MetricCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="bg-zinc-800/50 rounded-lg p-3">
      <div className="text-xs text-zinc-500">{label}</div>
      <div className="text-lg font-semibold mt-0.5">{value}</div>
      {sub && <div className="text-xs text-zinc-500 mt-0.5">{sub}</div>}
    </div>
  );
}

export function Recommendation({ architectures }: Props) {
  if (architectures.length === 0) {
    return (
      <div className="bg-zinc-800/50 border border-zinc-700 rounded-xl p-8 text-center">
        <div className="text-zinc-400 text-lg">No feasible architectures</div>
        <div className="text-zinc-500 text-sm mt-2">
          Try relaxing constraints or enabling more GPU/quantization options
        </div>
      </div>
    );
  }

  const best = architectures[0];
  const alternatives = architectures.slice(1, 6);

  return (
    <div className="space-y-6">
      {/* Best recommendation */}
      <div
        className={`relative rounded-xl border-2 p-6 ${
          best.meetsAll
            ? "border-emerald-500/50 bg-gradient-to-br from-emerald-950/30 to-zinc-900"
            : "border-amber-500/50 bg-gradient-to-br from-amber-950/30 to-zinc-900"
        }`}
      >
        <div className="absolute -top-3 left-4 px-3 py-0.5 bg-zinc-900 rounded-full text-xs font-semibold text-indigo-300 border border-indigo-500/40">
          RECOMMENDED
        </div>

        <div className="flex flex-col lg:flex-row lg:items-start gap-6 mt-2">
          {/* Left: Architecture */}
          <div className="flex-1 space-y-4">
            <div>
              <div className="text-xs text-zinc-500 uppercase tracking-wider">
                Architecture
              </div>
              <div className="text-xl font-bold mt-1">
                {best.disaggregated ? "Disaggregated" : "Unified"}{" "}
                <span className="text-indigo-400">[{best.quant}]</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-blue-900/20 border border-blue-800/40 rounded-lg p-3">
                <div className="text-xs text-blue-400 font-medium">
                  Prefill GPU
                </div>
                <div className="font-semibold mt-1">
                  {best.prefillGpuCount}x{" "}
                  {best.prefillGpu.gpuName.replace("NVIDIA ", "")}
                </div>
                <div className="text-xs text-zinc-400 mt-1">
                  ${best.prefillGpu.pricePerHour}/hr each
                </div>
                <div className="text-xs text-zinc-500">
                  {best.prefillGpu.prefillTps.toLocaleString()} tok/s
                  {best.prefillGpu.source === "measured" ? " *" : ""}
                </div>
              </div>
              <div className="bg-purple-900/20 border border-purple-800/40 rounded-lg p-3">
                <div className="text-xs text-purple-400 font-medium">
                  Decode GPU
                </div>
                <div className="font-semibold mt-1">
                  {best.decodeGpuCount}x{" "}
                  {best.decodeGpu.gpuName.replace("NVIDIA ", "")}
                </div>
                <div className="text-xs text-zinc-400 mt-1">
                  ${best.decodeGpu.pricePerHour}/hr each
                </div>
                <div className="text-xs text-zinc-500">
                  {best.decodeGpu.decodeTps.toLocaleString()} tok/s
                  {best.decodeGpu.source === "measured" ? " *" : ""}
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <Badge
                ok={best.meetsLatency}
                label="Latency"
                value={`${best.prefillLatencyMs.toFixed(0)}ms`}
              />
              <Badge
                ok={best.meetsBudget}
                label="Budget"
                value={`$${best.monthlySpend.toLocaleString()}/mo`}
              />
            </div>

            <p className="text-xs text-zinc-500 italic">{best.reason}</p>
          </div>

          {/* Right: Metrics */}
          <div className="grid grid-cols-2 gap-3 lg:w-72">
            <MetricCard
              label="Monthly Cost"
              value={`$${best.monthlySpend.toLocaleString()}`}
              sub={`$${best.hourlySpend.toFixed(0)}/hr`}
            />
            <MetricCard
              label="Prefill Latency"
              value={`${best.prefillLatencyMs.toFixed(1)}ms`}
              sub="time to first token"
            />
            <MetricCard
              label="Throughput"
              value={`${best.totalThroughputTps.toLocaleString()} t/s`}
              sub="total decode capacity"
            />
            <MetricCard
              label="Cost / 1K tokens"
              value={`$${best.costPer1kDecode.toFixed(4)}`}
              sub="decode phase"
            />
          </div>
        </div>
      </div>

      {/* Alternatives table */}
      {alternatives.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wider mb-3">
            Alternatives
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-zinc-500 uppercase border-b border-zinc-800">
                  <th className="pb-2 pr-4">Architecture</th>
                  <th className="pb-2 pr-4">Quant</th>
                  <th className="pb-2 pr-4 text-right">GPUs</th>
                  <th className="pb-2 pr-4 text-right">Cost/mo</th>
                  <th className="pb-2 pr-4 text-right">Latency</th>
                  <th className="pb-2 pr-4 text-right">Throughput</th>
                  <th className="pb-2 text-center">Meets?</th>
                </tr>
              </thead>
              <tbody>
                {alternatives.map((a, i) => (
                  <tr
                    key={a.id + i}
                    className="border-b border-zinc-800/50 hover:bg-zinc-800/30"
                  >
                    <td className="py-2.5 pr-4">
                      <div className="text-zinc-200">
                        {a.prefillGpu.gpuName.replace("NVIDIA ", "")}
                        {a.disaggregated &&
                        a.prefillGpu.gpuId !== a.decodeGpu.gpuId
                          ? ` + ${a.decodeGpu.gpuName.replace("NVIDIA ", "")}`
                          : ""}
                      </div>
                    </td>
                    <td className="py-2.5 pr-4">
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full ${
                          a.quant === "FP16"
                            ? "bg-blue-900/40 text-blue-300"
                            : a.quant === "INT8"
                            ? "bg-emerald-900/40 text-emerald-300"
                            : "bg-amber-900/40 text-amber-300"
                        }`}
                      >
                        {a.quant}
                      </span>
                    </td>
                    <td className="py-2.5 pr-4 text-right text-zinc-300">
                      {a.prefillGpuCount + a.decodeGpuCount}
                    </td>
                    <td className="py-2.5 pr-4 text-right text-zinc-300">
                      ${a.monthlySpend.toLocaleString()}
                    </td>
                    <td className="py-2.5 pr-4 text-right text-zinc-300">
                      {a.prefillLatencyMs.toFixed(0)}ms
                    </td>
                    <td className="py-2.5 pr-4 text-right text-zinc-300">
                      {a.totalThroughputTps.toLocaleString()} t/s
                    </td>
                    <td className="py-2.5 text-center">
                      {a.meetsAll ? (
                        <span className="text-emerald-400">{"\u2713"}</span>
                      ) : (
                        <span className="text-red-400">{"\u2717"}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
