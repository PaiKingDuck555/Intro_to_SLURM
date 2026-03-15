import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import * as fs from "fs";
import * as path from "path";

const execAsync = promisify(exec);

const SSH_HOST = process.env.BENCHMARK_SSH_HOST || "user04@35.84.33.219";
const SSH_KEY = process.env.BENCHMARK_SSH_KEY || "~/.ssh/id_ed25519";
const RESULTS_DIR = process.env.BENCHMARK_RESULTS_DIR ||
  path.join(process.cwd(), "..", "results");

async function sshCommand(cmd: string): Promise<string> {
  const { stdout } = await execAsync(
    `ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no ${SSH_HOST} "${cmd}"`,
    { timeout: 30000 }
  );
  return stdout.trim();
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { model, batch_size = 1, seq_length = 128, decode_tokens = 50 } = body;

    if (!model) {
      return NextResponse.json({ error: "model is required" }, { status: 400 });
    }

    // Submit a quick single-config benchmark via SSH
    const slurmScript = [
      "#!/bin/bash",
      `export PATH=$HOME/.local/bin:$PATH`,
      `python3 ~/disagg-bench/cluster/benchmark_inference.py \\`,
      `  --model "${model}" \\`,
      `  --batch-size ${batch_size} \\`,
      `  --seq-length ${seq_length} \\`,
      `  --decode-tokens ${decode_tokens} \\`,
      `  --dtype float16 \\`,
      `  --output-dir ~/disagg-bench/results`,
    ].join("\n");

    // Write temp script and submit
    const scriptName = `bench_quick_${Date.now()}.sh`;
    await sshCommand(`cat > /tmp/${scriptName} << 'SCRIPT'\n${slurmScript}\nSCRIPT`);

    const submitOutput = await sshCommand(
      `cd ~/disagg-bench && sbatch --job-name=quick_bench --output=logs/quick_%j.out ` +
      `--error=logs/quick_%j.err --ntasks=1 --cpus-per-task=4 --mem=64G ` +
      `--gres=gpu:1 --time=00:30:00 --partition=priority /tmp/${scriptName}`
    );

    const jobMatch = submitOutput.match(/Submitted batch job (\d+)/);
    if (!jobMatch) {
      return NextResponse.json(
        { error: "Failed to submit job", output: submitOutput },
        { status: 500 }
      );
    }

    const jobId = jobMatch[1];
    return NextResponse.json({ status: "submitted", jobId });
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
    // Check if job is still running
    const squeueOut = await sshCommand(`squeue -j ${jobId} --noheader 2>/dev/null || true`);

    if (squeueOut.trim().length > 0) {
      return NextResponse.json({ status: "running", jobId });
    }

    // Job finished — find the result file
    const resultFiles = await sshCommand(
      `ls -t ~/disagg-bench/results/bench_*.json 2>/dev/null | head -1`
    );

    if (!resultFiles) {
      return NextResponse.json({ status: "failed", error: "No result file found" });
    }

    const resultJson = await sshCommand(`cat ${resultFiles}`);
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
