"use client";

import { useState, useRef } from "react";
import { type ModelInfo } from "@/lib/benchmark-data";
import { MODEL_CATALOG, FAMILIES, type CatalogModel } from "@/lib/model-catalog";

interface Props {
  models: ModelInfo[];
  selected: string;
  onSelect: (id: string) => void;
  onRunBenchmark?: (modelId: string) => void;
}

export function ModelSelector({ models, selected, onSelect, onRunBenchmark }: Props) {
  const [familyFilter, setFamilyFilter] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const benchmarkedIds = new Set(models.map((m) => m.id));

  const filtered = familyFilter
    ? MODEL_CATALOG.filter((m) => m.family === familyFilter)
    : MODEL_CATALOG;

  const getBenchInfo = (cat: CatalogModel): ModelInfo | undefined =>
    models.find((m) => m.id === cat.id);

  return (
    <div className="space-y-4">
      {/* Family filter pills */}
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium text-zinc-400 uppercase tracking-wider mr-2">
          Models
        </label>
        <button
          onClick={() => setFamilyFilter(null)}
          className={`px-3 py-1 rounded-full text-xs font-medium border transition-all ${
            familyFilter === null
              ? "border-indigo-500 bg-indigo-500/15 text-indigo-300"
              : "border-zinc-700 bg-zinc-800/50 text-zinc-500 hover:border-zinc-500"
          }`}
        >
          All
        </button>
        {FAMILIES.map((f) => (
          <button
            key={f}
            onClick={() => setFamilyFilter(familyFilter === f ? null : f)}
            className={`px-3 py-1 rounded-full text-xs font-medium border transition-all ${
              familyFilter === f
                ? "border-indigo-500 bg-indigo-500/15 text-indigo-300"
                : "border-zinc-700 bg-zinc-800/50 text-zinc-500 hover:border-zinc-500"
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Scrollable model cards */}
      <div className="relative">
        <div
          ref={scrollRef}
          className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin"
          style={{ scrollbarWidth: "thin" }}
        >
          {filtered.map((cat) => {
            const hasBench = benchmarkedIds.has(cat.id);
            const benchInfo = getBenchInfo(cat);
            const isActive = selected === cat.id;

            return (
              <button
                key={cat.id}
                onClick={() => {
                  if (hasBench) onSelect(cat.id);
                }}
                className={`relative flex-shrink-0 w-52 p-4 rounded-xl border-2 text-left transition-all duration-200 ${
                  isActive
                    ? "border-indigo-500 bg-indigo-500/10 shadow-lg shadow-indigo-500/10"
                    : hasBench
                    ? "border-zinc-700 bg-zinc-800/50 hover:border-zinc-500 hover:bg-zinc-800 cursor-pointer"
                    : "border-zinc-800 bg-zinc-900/50 cursor-default opacity-70"
                }`}
              >
                {/* Status indicator */}
                {isActive && (
                  <div className="absolute top-3 right-3 w-2.5 h-2.5 rounded-full bg-indigo-400 animate-pulse" />
                )}
                {hasBench && !isActive && (
                  <div className="absolute top-3 right-3">
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-900/50 text-emerald-400 border border-emerald-800/40">
                      LIVE
                    </span>
                  </div>
                )}

                {/* Model info */}
                <div className="font-semibold text-base leading-tight">{cat.name}</div>
                <div className="text-xs text-zinc-500 mt-1">
                  {cat.params_b}B params &middot; {cat.fp16_gb} GB
                </div>

                {/* Quantization badges (only for benchmarked models) */}
                {hasBench && benchInfo && (
                  <div className="flex flex-wrap gap-1.5 mt-2.5">
                    {benchInfo.sizes.map((s) => (
                      <span
                        key={s.quant}
                        className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                          s.quant === "FP16"
                            ? "bg-blue-900/50 text-blue-300"
                            : s.quant === "INT8"
                            ? "bg-emerald-900/50 text-emerald-300"
                            : "bg-amber-900/50 text-amber-300"
                        }`}
                      >
                        {s.quant}
                      </span>
                    ))}
                  </div>
                )}

                {/* Run benchmark button for non-benchmarked models */}
                {!hasBench && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onRunBenchmark?.(cat.id);
                    }}
                    className="mt-3 w-full px-2 py-1.5 rounded-lg text-[11px] font-medium
                               border border-amber-700/50 bg-amber-900/20 text-amber-300
                               hover:bg-amber-900/40 hover:border-amber-600 transition-all"
                  >
                    Run Benchmark
                  </button>
                )}
              </button>
            );
          })}
        </div>

        {/* Scroll fade hints */}
        <div className="pointer-events-none absolute inset-y-0 right-0 w-12 bg-gradient-to-l from-[var(--bg)] to-transparent" />
      </div>
    </div>
  );
}
