import { NextRequest, NextResponse } from "next/server";
import { Client } from "ssh2";

const SSH_HOST = process.env.BENCHMARK_SSH_HOST || "35.84.33.219";
const SSH_USER = process.env.BENCHMARK_SSH_USER || "user04";
const SSH_PRIVATE_KEY = process.env.BENCHMARK_SSH_KEY || "";

function sshExec(cmd: string): Promise<string> {
  return new Promise((resolve, reject) => {
    if (!SSH_PRIVATE_KEY) {
      return reject(new Error("BENCHMARK_SSH_KEY environment variable not set"));
    }

    const conn = new Client();
    let output = "";
    let errorOutput = "";

    conn.on("ready", () => {
      conn.exec(cmd, (err, stream) => {
        if (err) {
          conn.end();
          return reject(err);
        }

        stream.on("data", (data: Buffer) => {
          output += data.toString();
        });

        stream.stderr.on("data", (data: Buffer) => {
          errorOutput += data.toString();
        });

        stream.on("close", (code: number) => {
          conn.end();
          if (code !== 0 && !output.trim()) {
            reject(new Error(errorOutput || `Command exited with code ${code}`));
          } else {
            resolve(output.trim());
          }
        });
      });
    });

    conn.on("error", (err) => {
      reject(new Error(`SSH connection failed: ${err.message}`));
    });

    conn.connect({
      host: SSH_HOST,
      port: 22,
      username: SSH_USER,
      privateKey: SSH_PRIVATE_KEY,
      readyTimeout: 15000,
    });
  });
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { model, batch_size = 1, seq_length = 128, decode_tokens = 50 } = body;

    if (!model) {
      return NextResponse.json({ error: "model is required" }, { status: 400 });
    }

    if (!SSH_PRIVATE_KEY) {
      return NextResponse.json(
        { error: "SSH key not configured. Set BENCHMARK_SSH_KEY in Vercel environment variables." },
        { status: 503 }
      );
    }

    const benchCmd = [
      "export PATH=$HOME/.local/bin:$PATH &&",
      "python3 ~/disagg-bench/cluster/benchmark_inference.py",
      `--model "${model}"`,
      `--batch-size ${batch_size}`,
      `--seq-length ${seq_length}`,
      `--decode-tokens ${decode_tokens}`,
      "--dtype float16",
      "--output-dir ~/disagg-bench/results",
    ].join(" ");

    // For quick benchmarks, run directly (not via SLURM) to get immediate results
    const output = await sshExec(benchCmd);

    // Parse the JSON result file — benchmark_inference.py prints the path
    const pathMatch = output.match(/Results saved to: (.+\.json)/);
    if (pathMatch) {
      const resultJson = await sshExec(`cat ${pathMatch[1]}`);
      const result = JSON.parse(resultJson);

      return NextResponse.json({
        status: "complete",
        result: {
          prefill_tps: result.prefill_tokens_per_sec,
          decode_tps: result.decode_tokens_per_sec,
          model_gb: result.model_size_gb,
          vram_prefill: result.vram_used_prefill_gb,
          vram_decode: result.vram_used_decode_gb,
          prefill_ms: result.prefill_time_ms,
          decode_ms_per_tok: result.decode_time_per_token_ms,
        },
      });
    }

    // If no path match, try to find the most recent result
    const latestFile = await sshExec(
      `ls -t ~/disagg-bench/results/bench_*.json 2>/dev/null | head -1`
    );

    if (latestFile) {
      const resultJson = await sshExec(`cat ${latestFile}`);
      const result = JSON.parse(resultJson);

      return NextResponse.json({
        status: "complete",
        result: {
          prefill_tps: result.prefill_tokens_per_sec,
          decode_tps: result.decode_tokens_per_sec,
          model_gb: result.model_size_gb,
          vram_prefill: result.vram_used_prefill_gb,
          vram_decode: result.vram_used_decode_gb,
          prefill_ms: result.prefill_time_ms,
          decode_ms_per_tok: result.decode_time_per_token_ms,
        },
      });
    }

    return NextResponse.json({
      status: "complete",
      raw_output: output,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    const isConnErr = msg.includes("ECONNREFUSED") || msg.includes("Connection refused") ||
      msg.includes("ssh") || msg.includes("timeout") || msg.includes("Permission denied");
    if (isConnErr) {
      return NextResponse.json(
        { error: "Cluster unavailable — benchmark execution requires a connected FluidStack GPU node. All benchmark data shown in the UI was collected on NVIDIA B200 via SLURM." },
        { status: 503 }
      );
    }
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

export async function GET(req: NextRequest) {
  const jobId = req.nextUrl.searchParams.get("jobId");
  if (!jobId) {
    return NextResponse.json({ error: "jobId required" }, { status: 400 });
  }

  try {
    const squeueOut = await sshExec(
      `squeue -j ${jobId} --noheader 2>/dev/null || true`
    );

    if (squeueOut.trim().length > 0) {
      return NextResponse.json({ status: "running", jobId });
    }

    const resultFiles = await sshExec(
      `ls -t ~/disagg-bench/results/bench_*.json 2>/dev/null | head -1`
    );

    if (!resultFiles) {
      return NextResponse.json({ status: "failed", error: "No result file found" });
    }

    const resultJson = await sshExec(`cat ${resultFiles}`);
    const result = JSON.parse(resultJson);

    return NextResponse.json({
      status: "complete",
      result: {
        prefill_tps: result.prefill_tokens_per_sec,
        decode_tps: result.decode_tokens_per_sec,
        model_gb: result.model_size_gb,
        vram_prefill: result.vram_used_prefill_gb,
        vram_decode: result.vram_used_decode_gb,
        prefill_ms: result.prefill_time_ms,
        decode_ms_per_tok: result.decode_time_per_token_ms,
      },
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
