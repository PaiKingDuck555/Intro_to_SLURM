"use client";

import { useState, useMemo, useCallback } from "react";
import { getModelList } from "@/lib/benchmark-data";
import { GPU_LIST } from "@/lib/gpu-catalog";
import { MODEL_CATALOG } from "@/lib/model-catalog";
import { optimize, type UserConstraints } from "@/lib/optimizer";
import { ModelSelector } from "@/components/model-selector";
import { ConstraintsPanel } from "@/components/constraints-panel";
import { Recommendation } from "@/components/recommendation";
import { ParetoChart } from "@/components/pareto-chart";
import { FineTunePlanner } from "@/components/finetune-planner";
import { BenchmarkModal } from "@/components/benchmark-modal";

const models = getModelList();

const DEFAULT_CONSTRAINTS: UserConstraints = {
  model: models[0]?.id || "",
  goal: "cost",
  maxLatencyMs: 200,
  maxMonthlyBudget: 50000,
  concurrentUsers: 100,
  avgPromptTokens: 256,
  avgOutputTokens: 128,
  allowedQuant: ["FP16", "INT8", "INT4"],
  allowedGpus: GPU_LIST.map((g) => g.id),
};

export default function Home() {
  const [constraints, setConstraints] =
    useState<UserConstraints>(DEFAULT_CONSTRAINTS);
  const [benchTarget, setBenchTarget] = useState<string | null>(null);

  const architectures = useMemo(() => optimize(constraints), [constraints]);

  const updateModel = (model: string) =>
    setConstraints((c) => ({ ...c, model }));

  const handleRunBenchmark = useCallback((modelId: string) => {
    setBenchTarget(modelId);
  }, []);

  const benchModelName =
    benchTarget
      ? MODEL_CATALOG.find((m) => m.id === benchTarget)?.name || benchTarget
      : "";

  return (
    <div className="min-h-screen">
      {/* Benchmark Modal */}
      {benchTarget && (
        <BenchmarkModal
          modelId={benchTarget}
          modelName={benchModelName}
          onClose={() => setBenchTarget(null)}
          onComplete={() => {
            setBenchTarget(null);
          }}
        />
      )}

      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm">
              I
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">InfraBench</h1>
              <p className="text-xs text-zinc-500">
                GPU Infrastructure Planner
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4 text-xs text-zinc-500">
            <span className="hidden sm:inline">
              Benchmarked on NVIDIA B200 &middot; FluidStack Cluster
            </span>
            <a
              href="https://github.com/damodarpai/Intro_to_SLURM"
              target="_blank"
              rel="noreferrer"
              className="px-3 py-1.5 rounded-md border border-zinc-700 hover:border-zinc-500 transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-10">
        {/* Hero */}
        <section className="text-center py-6">
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
            Find the optimal GPU architecture
            <br />
            <span className="text-indigo-400">for your LLM workload</span>
          </h2>
          <p className="text-zinc-400 mt-3 max-w-2xl mx-auto">
            Real benchmark data from NVIDIA B200, scaled across GPU families.
            Compare FP16, INT8, and INT4 quantization. Get cost-optimized
            disaggregated inference recommendations.
          </p>
        </section>

        {/* Step 1: Model Selection */}
        <section className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-5">
            <span className="flex items-center justify-center w-7 h-7 rounded-full bg-indigo-500/20 text-indigo-400 text-sm font-bold">
              1
            </span>
            <h2 className="text-lg font-semibold">Choose Your Model</h2>
            <span className="text-xs text-zinc-500 ml-auto">
              {models.length} benchmarked &middot; {MODEL_CATALOG.length - models.length} available to benchmark
            </span>
          </div>
          <ModelSelector
            models={models}
            selected={constraints.model}
            onSelect={updateModel}
            onRunBenchmark={handleRunBenchmark}
          />
        </section>

        {/* Step 2: Constraints */}
        <section className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-5">
            <span className="flex items-center justify-center w-7 h-7 rounded-full bg-indigo-500/20 text-indigo-400 text-sm font-bold">
              2
            </span>
            <h2 className="text-lg font-semibold">Define Constraints</h2>
          </div>
          <ConstraintsPanel
            constraints={constraints}
            onChange={setConstraints}
          />
        </section>

        {/* Step 3: Recommendation */}
        <section className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-5">
            <span className="flex items-center justify-center w-7 h-7 rounded-full bg-emerald-500/20 text-emerald-400 text-sm font-bold">
              3
            </span>
            <h2 className="text-lg font-semibold">Recommended Architecture</h2>
            <span className="text-xs text-zinc-500">
              {architectures.length} configurations evaluated
            </span>
          </div>
          <Recommendation architectures={architectures} />
        </section>

        {/* Pareto Chart */}
        <section className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6">
          <ParetoChart architectures={architectures} />
        </section>

        {/* Divider */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-zinc-800" />
          </div>
          <div className="relative flex justify-center">
            <span className="bg-[var(--bg)] px-4 text-sm text-zinc-500">
              Fine-Tuning
            </span>
          </div>
        </div>

        {/* Fine-Tuning Planner */}
        <section className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6">
          <FineTunePlanner models={models} />
        </section>

        {/* Footer */}
        <footer className="text-center py-8 text-xs text-zinc-600 space-y-1">
          <p>
            Built for the SemiAnalysis x FluidStack AI Infrastructure Hackathon
          </p>
          <p>
            Benchmark data collected on NVIDIA B200 (192 GB HBM3e) via SLURM
            cluster
          </p>
          <p>Performance estimates for other GPUs scaled by FLOPS and bandwidth ratios</p>
        </footer>
      </main>
    </div>
  );
}
