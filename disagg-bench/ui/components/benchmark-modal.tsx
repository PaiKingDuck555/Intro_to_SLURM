"use client";

import { useState, useCallback } from "react";

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

const SSH_CMD = (modelId: string) =>
  `python3 ~/disagg-bench/cluster/benchmark_inference.py \\\n  --model "${modelId}" \\\n  --batch-size 1 \\\n  --seq-length 128 \\\n  --decode-tokens 50 \\\n  --dtype float16 \\\n  --output-dir ~/disagg-bench/results`;

const SLURM_CMD = (modelId: string) =>
  `sbatch --job-name=bench_quick --output=logs/bench_%j.out \\
  --error=logs/bench_%j.err --ntasks=1 --cpus-per-task=4 \\
  --mem=64G --gres=gpu:1 --time=00:30:00 --partition=priority \\
  --wrap="python3 ~/disagg-bench/cluster/benchmark_inference.py \\
    --model \\"${modelId}\\" --batch-size 1 --seq-length 128 \\
    --decode-tokens 50 --dtype float16 \\
    --output-dir ~/disagg-bench/results"`;

export function BenchmarkModal({ modelId, modelName, onClose, onComplete }: Props) {
  const [status, setStatus] = useState<BenchStatus>("idle");
  const [log, setLog] = useState<string[]>([]);
  const [result, setResult] = useState<BenchResult | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [showLive, setShowLive] = useState(false);

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  }, []);

  const copyToClipboard = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
      setTimeout(() => setCopied(null), 2000);
    } catch {
      setCopied("failed");
    }
  };

  const runLiveBenchmark = async () => {
    setShowLive(true);
    setStatus("submitting");
    setLog([]);
    addLog(`Connecting to cluster...`);
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
      setStatus("error");
      if (msg.includes("ssh") || msg.includes("command not found") || msg.includes("SSH")) {
        addLog("SSH not available in this environment.");
        addLog("This feature requires running InfraBench locally.");
        addLog("Use the commands above to run manually.");
      } else {
        addLog(`Error: ${msg}`);
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

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-zinc-900 border border-zinc-700 rounded-2xl w-full max-w-2xl mx-4 shadow-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-zinc-800 sticky top-0 bg-zinc-900 rounded-t-2xl z-10">
          <div>
            <h3 className="font-semibold text-lg">Benchmark {modelName}</h3>
            <p className="text-xs text-zinc-500 mt-0.5">Run on your NVIDIA B200 cluster via SLURM</p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-zinc-800 hover:bg-zinc-700 flex items-center justify-center text-zinc-400 transition-colors"
          >
            &times;
          </button>
        </div>

        <div className="p-5 space-y-5">
          {/* Step 1: SSH into cluster */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 text-xs font-bold">1</span>
              <span className="text-sm font-medium text-zinc-300">SSH into your FluidStack cluster</span>
            </div>
            <div className="relative">
              <pre className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 text-xs font-mono text-zinc-300 overflow-x-auto">
                ssh -i ~/.ssh/id_ed25519 user04@35.84.33.219
              </pre>
              <button
                onClick={() => copyToClipboard("ssh -i ~/.ssh/id_ed25519 user04@35.84.33.219", "ssh")}
                className="absolute top-2 right-2 px-2 py-1 rounded text-[10px] font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-400 transition-colors"
              >
                {copied === "ssh" ? "Copied!" : "Copy"}
              </button>
            </div>
          </div>

          {/* Step 2: Run directly or via SLURM */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 text-xs font-bold">2</span>
              <span className="text-sm font-medium text-zinc-300">Run the benchmark</span>
            </div>

            <div className="space-y-3">
              {/* Direct run */}
              <div>
                <div className="text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Option A: Direct (interactive)</div>
                <div className="relative">
                  <pre className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 text-xs font-mono text-emerald-300/80 overflow-x-auto whitespace-pre-wrap">
                    {SSH_CMD(modelId)}
                  </pre>
                  <button
                    onClick={() => copyToClipboard(SSH_CMD(modelId).replace(/\\\n/g, ""), "direct")}
                    className="absolute top-2 right-2 px-2 py-1 rounded text-[10px] font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-400 transition-colors"
                  >
                    {copied === "direct" ? "Copied!" : "Copy"}
                  </button>
                </div>
              </div>

              {/* SLURM submit */}
              <div>
                <div className="text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Option B: SLURM batch job</div>
                <div className="relative">
                  <pre className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 text-xs font-mono text-amber-300/80 overflow-x-auto whitespace-pre-wrap">
                    {SLURM_CMD(modelId)}
                  </pre>
                  <button
                    onClick={() => copyToClipboard(SLURM_CMD(modelId).replace(/\\\n/g, ""), "slurm")}
                    className="absolute top-2 right-2 px-2 py-1 rounded text-[10px] font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-400 transition-colors"
                  >
                    {copied === "slurm" ? "Copied!" : "Copy"}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Step 3: Results */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 text-xs font-bold">3</span>
              <span className="text-sm font-medium text-zinc-300">Results appear in <code className="text-xs bg-zinc-800 px-1.5 py-0.5 rounded">~/disagg-bench/results/</code></span>
            </div>
            <p className="text-xs text-zinc-500 ml-7">
              The benchmark outputs a JSON file with prefill/decode tokens-per-second, VRAM usage, and latency data.
              Add the results to <code className="bg-zinc-800 px-1 py-0.5 rounded">lib/benchmark-data.ts</code> to include this model in the optimizer.
            </p>
          </div>

          {/* Live connection attempt (expandable) */}
          <div className="border-t border-zinc-800 pt-4">
            <button
              onClick={() => {
                if (!showLive) runLiveBenchmark();
                else setShowLive(!showLive);
              }}
              className="flex items-center gap-2 text-sm text-zinc-400 hover:text-zinc-300 transition-colors"
            >
              <span className="text-xs">{showLive ? "\u25BC" : "\u25B6"}</span>
              <span>Try live connection</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-500">localhost only</span>
            </button>

            {showLive && (
              <div className="mt-3">
                {/* Status bar */}
                <div className="flex items-center gap-3 mb-2">
                  {(status === "submitting" || status === "running" || status === "polling") && (
                    <div className="w-3 h-3 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                  )}
                  {status === "complete" && (
                    <div className="w-3 h-3 rounded-full bg-emerald-500" />
                  )}
                  {status === "error" && (
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                  )}
                  <span className={`text-xs font-medium ${
                    status === "complete" ? "text-emerald-400"
                    : status === "error" ? "text-red-400"
                    : "text-indigo-300"
                  }`}>
                    {status === "submitting" && "Connecting..."}
                    {status === "running" && "Running on B200..."}
                    {status === "polling" && "Waiting for results..."}
                    {status === "complete" && "Complete"}
                    {status === "error" && "Connection failed"}
                  </span>
                </div>

                {/* Log output */}
                <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 max-h-32 overflow-y-auto">
                  <div className="font-mono text-[11px] space-y-0.5">
                    {log.map((line, i) => (
                      <div key={i} className={
                        line.includes("Error") || line.includes("not available")
                          ? "text-red-400"
                          : line.includes("complete") ? "text-emerald-400"
                          : "text-zinc-500"
                      }>
                        {line}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Result cards */}
                {result && (
                  <div className="grid grid-cols-3 gap-3 mt-3">
                    <div className="bg-zinc-800 rounded-lg p-3 text-center">
                      <div className="text-xs text-zinc-500">Prefill</div>
                      <div className="text-lg font-bold text-blue-400">{result.prefill_tps.toLocaleString()}</div>
                      <div className="text-[10px] text-zinc-500">tok/s</div>
                    </div>
                    <div className="bg-zinc-800 rounded-lg p-3 text-center">
                      <div className="text-xs text-zinc-500">Decode</div>
                      <div className="text-lg font-bold text-purple-400">{result.decode_tps.toLocaleString()}</div>
                      <div className="text-[10px] text-zinc-500">tok/s</div>
                    </div>
                    <div className="bg-zinc-800 rounded-lg p-3 text-center">
                      <div className="text-xs text-zinc-500">VRAM</div>
                      <div className="text-lg font-bold text-amber-400">{result.model_gb.toFixed(1)}</div>
                      <div className="text-[10px] text-zinc-500">GB FP16</div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-5 border-t border-zinc-800">
          {result && (
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
