"use client";

import { useState } from "react";
import {
  estimateFineTune,
  type FineTuneEstimate,
} from "@/lib/optimizer";
import type { ModelInfo } from "@/lib/benchmark-data";

interface Props {
  models: ModelInfo[];
}

export function FineTunePlanner({ models }: Props) {
  const [selectedModel, setSelectedModel] = useState(models[0]?.id || "");
  const [datasetSize, setDatasetSize] = useState(1_000_000);
  const [epochs, setEpochs] = useState(3);
  const [method, setMethod] = useState<"full" | "lora" | "qlora">("lora");
  const [contextLength, setContextLength] = useState(2048);
  const [maxBudget, setMaxBudget] = useState(5000);

  const model = models.find((m) => m.id === selectedModel);
  const fp16Size = model?.sizes.find((s) => s.quant === "FP16")?.gb || 0;

  let estimate: FineTuneEstimate | null = null;
  if (model && fp16Size > 0) {
    estimate = estimateFineTune(
      model.id,
      model.params_b,
      fp16Size,
      datasetSize,
      epochs,
      method,
      contextLength,
      maxBudget
    );
  }

  const methods = [
    {
      value: "full" as const,
      label: "Full Fine-tune",
      desc: "All weights updated",
    },
    { value: "lora" as const, label: "LoRA", desc: "Low-rank adapters" },
    {
      value: "qlora" as const,
      label: "QLoRA",
      desc: "Quantized + LoRA",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold">Fine-Tuning Planner</h2>
        <span className="text-xs px-2 py-0.5 rounded-full bg-amber-900/50 text-amber-300 border border-amber-700/50">
          BETA
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Inputs */}
        <div className="space-y-5 bg-zinc-800/30 border border-zinc-700/50 rounded-xl p-5">
          {/* Model */}
          <div>
            <label className="block text-xs text-zinc-400 mb-1.5">
              Base Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.params_b}B)
                </option>
              ))}
            </select>
          </div>

          {/* Method */}
          <div>
            <label className="block text-xs text-zinc-400 mb-2">
              Training Method
            </label>
            <div className="grid grid-cols-3 gap-2">
              {methods.map((m) => (
                <button
                  key={m.value}
                  onClick={() => setMethod(m.value)}
                  className={`px-3 py-2.5 rounded-lg border text-sm text-left transition-all ${
                    method === m.value
                      ? "border-indigo-500 bg-indigo-500/15 text-indigo-300"
                      : "border-zinc-700 bg-zinc-800/50 text-zinc-400 hover:border-zinc-500"
                  }`}
                >
                  <div className="font-medium">{m.label}</div>
                  <div className="text-xs opacity-60 mt-0.5">{m.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Numeric inputs */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Dataset Size
              </label>
              <div className="relative">
                <input
                  type="number"
                  value={datasetSize}
                  onChange={(e) => setDatasetSize(Number(e.target.value))}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                             focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-zinc-500">
                  tokens
                </span>
              </div>
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Epochs
              </label>
              <input
                type="number"
                value={epochs}
                min={1}
                max={100}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Context Length
              </label>
              <div className="relative">
                <input
                  type="number"
                  value={contextLength}
                  onChange={(e) => setContextLength(Number(e.target.value))}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                             focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-zinc-500">
                  tokens
                </span>
              </div>
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">
                Max Budget
              </label>
              <div className="relative">
                <input
                  type="number"
                  value={maxBudget}
                  onChange={(e) => setMaxBudget(Number(e.target.value))}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                             focus:outline-none focus:ring-2 focus:ring-indigo-500 text-zinc-100"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-zinc-500">
                  $
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Results */}
        <div className="bg-zinc-800/30 border border-zinc-700/50 rounded-xl p-5">
          {estimate ? (
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wider">
                Estimated Training Plan
              </h3>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-zinc-800/80 rounded-lg p-3">
                  <div className="text-xs text-zinc-500">GPU Recommendation</div>
                  <div className="text-lg font-semibold mt-1">
                    {estimate.gpuRecommendation}
                  </div>
                </div>
                <div className="bg-zinc-800/80 rounded-lg p-3">
                  <div className="text-xs text-zinc-500">VRAM Required</div>
                  <div className="text-lg font-semibold mt-1">
                    {estimate.vramRequired} GB
                  </div>
                </div>
                <div className="bg-zinc-800/80 rounded-lg p-3">
                  <div className="text-xs text-zinc-500">Estimated Time</div>
                  <div className="text-lg font-semibold mt-1">
                    {estimate.estimatedHours < 1
                      ? `${(estimate.estimatedHours * 60).toFixed(0)} min`
                      : `${estimate.estimatedHours} hrs`}
                  </div>
                </div>
                <div className="bg-zinc-800/80 rounded-lg p-3">
                  <div className="text-xs text-zinc-500">Estimated Cost</div>
                  <div
                    className={`text-lg font-semibold mt-1 ${
                      estimate.estimatedCost <= maxBudget
                        ? "text-emerald-400"
                        : "text-red-400"
                    }`}
                  >
                    ${estimate.estimatedCost.toLocaleString()}
                  </div>
                </div>
              </div>

              <div className="bg-zinc-800/80 rounded-lg p-3">
                <div className="text-xs text-zinc-500">Checkpoint Size</div>
                <div className="text-sm font-medium mt-1">
                  {estimate.checkpointSizeGb} GB per checkpoint
                </div>
                <div className="text-xs text-zinc-500 mt-1">
                  Method: {method.toUpperCase()} &middot; Epochs: {epochs}{" "}
                  &middot; Total tokens:{" "}
                  {(datasetSize * epochs).toLocaleString()}
                </div>
              </div>

              <div className="text-xs text-zinc-500 bg-zinc-900/50 rounded-lg p-3 border border-zinc-700/50">
                <span className="text-amber-400 font-medium">Note:</span>{" "}
                These are estimates based on hardware scaling ratios. Actual
                training times depend on model architecture, optimizer settings,
                data loading, and communication overhead. Run a benchmark on
                your cluster for precise numbers.
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-zinc-500 text-sm">
              Select a model to see training estimates
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
