"use client";

import { useState, useEffect, useCallback } from "react";

interface Props {
  modelId: string;
  modelName: string;
  onClose: () => void;
  onComplete: () => void;
}

type BenchStatus = "idle" | "submitting" | "running" | "polling" | "complete" | "error";

interface BenchResult {
  prefill_tps: number;
  decode_tps: number;
  model_gb: number;
  vram_prefill: number;
  vram_decode: number;
  prefill_ms: number;
  decode_ms_per_tok: number;
}

export function BenchmarkModal({ modelId, modelName, onClose, onComplete }: Props) {
  const [status, setStatus] = useState<BenchStatus>("idle");
  const [log, setLog] = useState<string[]>([]);
  const [result, setResult] = useState<BenchResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  }, []);

  const runBenchmark = async () => {
    setStatus("submitting");
    addLog(`Submitting benchmark for ${modelName}...`);
    addLog(`Model: ${modelId}`);
    addLog(`Config: batch=1, seq=128, decode_tokens=50`);

    try {
      const res = await fetch("/api/benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: modelId,
          batch_size: 1,
          seq_length: 128,
          decode_tokens: 50,
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const data = await res.json();

      if (data.status === "submitted") {
        setStatus("running");
        addLog(`SLURM job submitted: ${data.jobId}`);
        addLog("Waiting for results from B200...");
        pollForResults(data.jobId);
      } else if (data.status === "complete") {
        handleResult(data.result);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      setStatus("error");
      addLog(`Error: ${msg}`);

      if (msg.includes("fetch") || msg.includes("Failed") || msg.includes("ECONNREFUSED")) {
        addLog("");
        addLog("Cannot reach benchmark API. To run live benchmarks:");
        addLog("1. SSH into your FluidStack cluster");
        addLog(`2. Run: python3 ~/disagg-bench/cluster/benchmark_inference.py \\`);
        addLog(`     --model ${modelId} --batch-size 1 --seq-length 128 \\`);
        addLog(`     --decode-tokens 50 --output-dir ~/disagg-bench/results`);
        addLog("3. Download the result JSON and add to the UI");
      }
    }
  };

  const pollForResults = async (jobId: string) => {
    setStatus("polling");
    let attempts = 0;
    const maxAttempts = 60;

    const poll = async () => {
      attempts++;
      try {
        const res = await fetch(`/api/benchmark?jobId=${jobId}`);
        const data = await res.json();

        if (data.status === "complete") {
          handleResult(data.result);
          return;
        } else if (data.status === "failed") {
          throw new Error("Benchmark job failed");
        } else {
          if (attempts % 5 === 0) addLog(`Still running... (${attempts * 5}s)`);
          if (attempts < maxAttempts) {
            setTimeout(poll, 5000);
          } else {
            throw new Error("Timeout waiting for results");
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Poll error";
        setError(msg);
        setStatus("error");
        addLog(`Error: ${msg}`);
      }
    };

    poll();
  };

  const handleResult = (r: BenchResult) => {
    setResult(r);
    setStatus("complete");
    addLog("Benchmark complete!");
    addLog(`Prefill: ${r.prefill_tps.toLocaleString()} tokens/sec`);
    addLog(`Decode: ${r.decode_tps.toLocaleString()} tokens/sec`);
    addLog(`VRAM: ${r.vram_prefill.toFixed(1)} GB`);
    addLog(`Model size: ${r.model_gb.toFixed(1)} GB FP16`);
  };

  useEffect(() => {
    if (status === "idle") {
      runBenchmark();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-zinc-900 border border-zinc-700 rounded-2xl w-full max-w-xl mx-4 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-zinc-800">
          <div>
            <h3 className="font-semibold text-lg">Run Benchmark</h3>
            <p className="text-xs text-zinc-500 mt-0.5">{modelName} on NVIDIA B200</p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-zinc-800 hover:bg-zinc-700 flex items-center justify-center text-zinc-400 transition-colors"
          >
            &times;
          </button>
        </div>

        {/* Status bar */}
        <div className="px-5 py-3 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            {(status === "submitting" || status === "running" || status === "polling") && (
              <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
            )}
            {status === "complete" && (
              <div className="w-4 h-4 rounded-full bg-emerald-500 flex items-center justify-center text-[10px] text-white">
                {"\u2713"}
              </div>
            )}
            {status === "error" && (
              <div className="w-4 h-4 rounded-full bg-red-500 flex items-center justify-center text-[10px] text-white">
                !
              </div>
            )}
            <span className={`text-sm font-medium ${
              status === "complete" ? "text-emerald-400"
              : status === "error" ? "text-red-400"
              : "text-indigo-300"
            }`}>
              {status === "submitting" && "Submitting to SLURM..."}
              {status === "running" && "Running on B200..."}
              {status === "polling" && "Waiting for results..."}
              {status === "complete" && "Benchmark complete"}
              {status === "error" && "Benchmark failed"}
            </span>
          </div>
        </div>

        {/* Log */}
        <div className="p-4 max-h-64 overflow-y-auto">
          <div className="font-mono text-xs space-y-0.5">
            {log.map((line, i) => (
              <div
                key={i}
                className={
                  line.startsWith("[") && line.includes("Error")
                    ? "text-red-400"
                    : line.startsWith("[") && line.includes("complete")
                    ? "text-emerald-400"
                    : "text-zinc-400"
                }
              >
                {line}
              </div>
            ))}
          </div>
        </div>

        {/* Result summary */}
        {result && (
          <div className="px-5 pb-4">
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-zinc-800 rounded-lg p-3 text-center">
                <div className="text-xs text-zinc-500">Prefill</div>
                <div className="text-lg font-bold text-blue-400">
                  {result.prefill_tps.toLocaleString()}
                </div>
                <div className="text-[10px] text-zinc-500">tok/s</div>
              </div>
              <div className="bg-zinc-800 rounded-lg p-3 text-center">
                <div className="text-xs text-zinc-500">Decode</div>
                <div className="text-lg font-bold text-purple-400">
                  {result.decode_tps.toLocaleString()}
                </div>
                <div className="text-[10px] text-zinc-500">tok/s</div>
              </div>
              <div className="bg-zinc-800 rounded-lg p-3 text-center">
                <div className="text-xs text-zinc-500">VRAM</div>
                <div className="text-lg font-bold text-amber-400">
                  {result.model_gb.toFixed(1)}
                </div>
                <div className="text-[10px] text-zinc-500">GB FP16</div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="flex justify-end gap-3 p-5 border-t border-zinc-800">
          {status === "error" && (
            <button
              onClick={() => { setStatus("idle"); setError(null); setLog([]); runBenchmark(); }}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
            >
              Retry
            </button>
          )}
          {status === "complete" && (
            <button
              onClick={() => { onComplete(); onClose(); }}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-emerald-600 hover:bg-emerald-500 text-white transition-colors"
            >
              Use Results
            </button>
          )}
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-sm font-medium border border-zinc-700 hover:border-zinc-500 text-zinc-300 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
