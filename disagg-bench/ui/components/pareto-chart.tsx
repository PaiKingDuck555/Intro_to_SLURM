"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { Architecture } from "@/lib/optimizer";

interface Props {
  architectures: Architecture[];
}

const COLORS = {
  meets: "#4ade80",
  fails: "#f87171",
  best: "#818cf8",
};

interface ChartDatum {
  x: number;
  y: number;
  label: string;
  quant: string;
  meetsAll: boolean;
  isBest: boolean;
}

export function ParetoChart({ architectures }: Props) {
  if (architectures.length === 0) return null;

  const data: ChartDatum[] = architectures.slice(0, 20).map((a, i) => ({
    x: a.monthlySpend,
    y: a.prefillLatencyMs,
    label: `${a.prefillGpu.gpuName.replace("NVIDIA ", "")}${
      a.disaggregated && a.prefillGpu.gpuId !== a.decodeGpu.gpuId
        ? ` + ${a.decodeGpu.gpuName.replace("NVIDIA ", "")}`
        : ""
    } [${a.quant}]`,
    quant: a.quant,
    meetsAll: a.meetsAll,
    isBest: i === 0,
  }));

  return (
    <div>
      <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wider mb-4">
        Cost vs Latency Tradeoff
      </h3>
      <div className="bg-zinc-800/30 border border-zinc-700/50 rounded-xl p-4">
        <ResponsiveContainer width="100%" height={320}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis
              dataKey="x"
              type="number"
              name="Monthly Cost"
              tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
              stroke="#52525b"
              fontSize={12}
              label={{
                value: "Monthly Cost ($)",
                position: "insideBottom",
                offset: -10,
                style: { fill: "#71717a", fontSize: 12 },
              }}
            />
            <YAxis
              dataKey="y"
              type="number"
              name="Latency"
              tickFormatter={(v) => `${v}ms`}
              stroke="#52525b"
              fontSize={12}
              label={{
                value: "Prefill Latency (ms)",
                angle: -90,
                position: "insideLeft",
                offset: 0,
                style: { fill: "#71717a", fontSize: 12 },
              }}
            />
            <Tooltip
              content={({ payload }) => {
                if (!payload?.length) return null;
                const d = payload[0].payload as ChartDatum;
                return (
                  <div className="bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-xs shadow-xl">
                    <div className="font-semibold text-zinc-200">{d.label}</div>
                    <div className="text-zinc-400 mt-1">
                      Cost: ${d.x.toLocaleString()}/mo
                    </div>
                    <div className="text-zinc-400">
                      Latency: {d.y.toFixed(1)}ms
                    </div>
                    <div
                      className={
                        d.meetsAll ? "text-emerald-400" : "text-red-400"
                      }
                    >
                      {d.meetsAll
                        ? "Meets all constraints"
                        : "Fails constraints"}
                    </div>
                  </div>
                );
              }}
            />
            <Scatter data={data}>
              {data.map((d, i) => (
                <Cell
                  key={i}
                  fill={
                    d.isBest
                      ? COLORS.best
                      : d.meetsAll
                      ? COLORS.meets
                      : COLORS.fails
                  }
                  r={d.isBest ? 8 : 5}
                  strokeWidth={d.isBest ? 2 : 0}
                  stroke={d.isBest ? "#c7d2fe" : "none"}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
        <div className="flex justify-center gap-6 mt-2 text-xs text-zinc-500">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-indigo-400 inline-block" />{" "}
            Best
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-emerald-400 inline-block" />{" "}
            Feasible
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-red-400 inline-block" />{" "}
            Fails constraints
          </span>
        </div>
      </div>
    </div>
  );
}
