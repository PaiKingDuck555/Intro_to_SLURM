"use client";

import { useState, useEffect, useCallback } from "react";

interface Props {
  modelId: string;
  modelName: string;
  onClose: () => void;
  onComplete: () => void;
}

type BenchStatus = "idle" | "connecting" | "running" | "complete" | "needs-key" | "error";

interface BenchResult {
  prefill_tps: number;
  decode_tps: number;
  model_gb: number;
  vram_prefill: number;
  vram_decode: number;
  prefill_ms: number;
  decode_ms_per_tok: number;
}

const MANUAL_CMD = (modelId: string) =>
  `python3 ~/disagg-bench/cluster/benchmark_inference.py \\\n  --model "${modelId}" \\\n  --batch-size 1 --seq-length 128 \\\n  --decode-tokens 50 --dtype float16 \\\n  --output-dir ~/disagg-bench/results`;

export function BenchmarkModal({ modelId, modelName, onClose, onComplete }: Props) {
  const [status, setStatus] = useState<BenchStatus>("idle");
  const [log, setLog] = useState<string[]>([]);
  const [result, setResult] = useState<BenchResult | null>(null);
  const [copied, setCopied] = useState(false);

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  }, []);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* ignore */ }
  };

  const runBenchmark = useCallback(async () => {
    setStatus("connecting");
    setLog([]);
    addLog(`Connecting to B200 cluster via SSH...`);
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

      const data = await res.json();

      if (!res.ok) {
        if (res.status === 503 || (data.error && data.error.includes("SSH key"))) {
          setStatus("needs-key");
          addLog("SSH key not configured on server.");
          return;
        }
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      if (data.status === "complete" && data.result) {
        setResult(data.result);
        setStatus("complete");
        addLog("Connected. Benchmark running on GPU...");
        addLog("Benchmark complete!");
        addLog(`Prefill: ${data.result.prefill_tps.toLocaleString()} tok/s`);
        addLog(`Decode:  ${data.result.decode_tps.toLocaleString()} tok/s`);
        addLog(`VRAM:    ${data.result.model_gb.toFixed(1)} GB`);
      } else {
        throw new Error("Unexpected response format");
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setStatus("error");
      addLog(`Error: ${msg}`);
    }
  }, [modelId, addLog]);

  useEffect(() => {
    runBenchmark();
  }, [runBenchmark]);

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-zinc-900 border border-zinc-700 rounded-2xl w-full max-w-xl mx-4 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-zinc-800">
          <div>
            <h3 className="font-semibold text-lg">Benchmark {modelName}</h3>
            <p className="text-xs text-zinc-500 mt-0.5">NVIDIA B200 &middot; FluidStack Cluster</p>
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
            {(status === "connecting" || status === "running") && (
              <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
            )}
            {status === "complete" && (
              <div className="w-4 h-4 rounded-full bg-emerald-500 flex items-center justify-center text-[10px] text-white">{"\u2713"}</div>
            )}
            {status === "needs-key" && (
              <div className="w-4 h-4 rounded-full bg-amber-500 flex items-center justify-center text-[10px] text-white">!</div>
            )}
            {status === "error" && (
              <div className="w-4 h-4 rounded-full bg-red-500 flex items-center justify-center text-[10px] text-white">!</div>
            )}
            <span className={`text-sm font-medium ${
              status === "complete" ? "text-emerald-400"
              : status === "needs-key" ? "text-amber-400"
              : status === "error" ? "text-red-400"
              : "text-indigo-300"
            }`}>
              {status === "connecting" && "Connecting to cluster..."}
              {status === "running" && "Running benchmark on B200..."}
              {status === "complete" && "Benchmark complete"}
              {status === "needs-key" && "SSH key required"}
              {status === "error" && "Benchmark failed"}
            </span>
          </div>
        </div>

        {/* Main content */}
        <div className="p-5 space-y-4">
          {/* Needs SSH key setup */}
          {status === "needs-key" && (
            <div className="space-y-3">
              <p className="text-sm text-zinc-300">
                To enable live benchmarking, add your SSH private key as a Vercel environment variable:
              </p>
              <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 space-y-2 text-xs font-mono">
                <div className="text-amber-300">
                  Variable: <span className="text-zinc-300">BENCHMARK_SSH_KEY</span>
                </div>
                <div className="text-amber-300">
                  Value: <span className="text-zinc-400">contents of ~/.ssh/id_ed25519</span>
                </div>
              </div>
              <p className="text-xs text-zinc-500">
                Vercel Dashboard &rarr; Settings &rarr; Environment Variables &rarr; Add <code className="bg-zinc-800 px-1 rounded">BENCHMARK_SSH_KEY</code>
              </p>
              <div className="border-t border-zinc-800 pt-3">
                <p className="text-xs text-zinc-400 mb-2">Or run manually on your cluster:</p>
                <div className="relative">
                  <pre className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 text-xs font-mono text-emerald-300/80 overflow-x-auto whitespace-pre-wrap">
                    {MANUAL_CMD(modelId)}
                  </pre>
                  <button
                    onClick={() => copyToClipboard(MANUAL_CMD(modelId).replace(/\\\n\s*/g, " "))}
                    className="absolute top-2 right-2 px-2 py-1 rounded text-[10px] font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-400 transition-colors"
                  >
                    {copied ? "Copied!" : "Copy"}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Log (shown during connecting/running/error/complete) */}
          {status !== "needs-key" && (
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 max-h-48 overflow-y-auto">
              <div className="font-mono text-xs space-y-0.5">
                {log.map((line, i) => (
                  <div key={i} className={
                    line.includes("Error") ? "text-red-400"
                    : line.includes("complete") ? "text-emerald-400"
                    : line.includes("Prefill") || line.includes("Decode") || line.includes("VRAM")
                    ? "text-indigo-300"
                    : "text-zinc-500"
                  }>
                    {line}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Result cards */}
          {result && (
            <div className="grid grid-cols-3 gap-3">
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

          {/* Error fallback with manual command */}
          {status === "error" && (
            <div className="border-t border-zinc-800 pt-3">
              <p className="text-xs text-zinc-400 mb-2">Run manually on your cluster:</p>
              <div className="relative">
                <pre className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 text-xs font-mono text-emerald-300/80 overflow-x-auto whitespace-pre-wrap">
                  {MANUAL_CMD(modelId)}
                </pre>
                <button
                  onClick={() => copyToClipboard(MANUAL_CMD(modelId).replace(/\\\n\s*/g, " "))}
                  className="absolute top-2 right-2 px-2 py-1 rounded text-[10px] font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-400 transition-colors"
                >
                  {copied ? "Copied!" : "Copy"}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-5 border-t border-zinc-800">
          {status === "error" && (
            <button
              onClick={runBenchmark}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
            >
              Retry
            </button>
          )}
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
