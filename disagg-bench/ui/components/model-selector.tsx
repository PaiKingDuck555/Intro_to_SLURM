"use client";

import { type ModelInfo } from "@/lib/benchmark-data";

interface Props {
  models: ModelInfo[];
  selected: string;
  onSelect: (id: string) => void;
}

export function ModelSelector({ models, selected, onSelect }: Props) {
  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-zinc-400 uppercase tracking-wider">
        Select Model
      </label>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {models.map((m) => {
          const isActive = selected === m.id;
          return (
            <button
              key={m.id}
              onClick={() => onSelect(m.id)}
              className={`relative p-4 rounded-xl border-2 text-left transition-all duration-200 ${
                isActive
                  ? "border-indigo-500 bg-indigo-500/10 shadow-lg shadow-indigo-500/10"
                  : "border-zinc-700 bg-zinc-800/50 hover:border-zinc-500 hover:bg-zinc-800"
              }`}
            >
              {isActive && (
                <div className="absolute top-3 right-3 w-2.5 h-2.5 rounded-full bg-indigo-400 animate-pulse" />
              )}
              <div className="font-semibold text-lg">{m.name}</div>
              <div className="text-sm text-zinc-400 mt-1">
                {m.params_b}B parameters
              </div>
              <div className="flex flex-wrap gap-2 mt-3">
                {m.sizes.map((s) => (
                  <span
                    key={s.quant}
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      s.quant === "FP16"
                        ? "bg-blue-900/50 text-blue-300"
                        : s.quant === "INT8"
                        ? "bg-emerald-900/50 text-emerald-300"
                        : "bg-amber-900/50 text-amber-300"
                    }`}
                  >
                    {s.quant}: {s.gb.toFixed(1)} GB
                  </span>
                ))}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
