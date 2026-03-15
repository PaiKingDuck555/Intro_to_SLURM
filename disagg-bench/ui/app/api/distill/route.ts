import { NextRequest, NextResponse } from "next/server";
import { Client } from "ssh2";

const SSH_HOST = process.env.BENCHMARK_SSH_HOST || "35.84.33.219";
const SSH_USER = process.env.BENCHMARK_SSH_USER || "user04";
const SSH_PRIVATE_KEY = process.env.BENCHMARK_SSH_KEY || "";

function sshExec(cmd: string, timeoutMs = 1800000): Promise<string> {
  return new Promise((resolve, reject) => {
    if (!SSH_PRIVATE_KEY) {
      return reject(new Error("BENCHMARK_SSH_KEY not set. Add your SSH private key in Vercel environment variables."));
    }

    const conn = new Client();
    let output = "";
    let errorOutput = "";

    const timer = setTimeout(() => {
      conn.end();
      reject(new Error("SSH command timed out"));
    }, timeoutMs);

    conn.on("ready", () => {
      conn.exec(cmd, (err, stream) => {
        if (err) { clearTimeout(timer); conn.end(); return reject(err); }

        stream.on("data", (data: Buffer) => { output += data.toString(); });
        stream.stderr.on("data", (data: Buffer) => { errorOutput += data.toString(); });

        stream.on("close", (code: number) => {
          clearTimeout(timer);
          conn.end();
          if (code !== 0 && !output.trim()) {
            reject(new Error(errorOutput.slice(0, 500) || `Exit code ${code}`));
          } else {
            resolve(output.trim());
          }
        });
      });
    });

    conn.on("error", (err) => {
      clearTimeout(timer);
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
    const {
      teacher,
      student,
      dataset = "databricks/databricks-dolly-15k",
      numSamples = 5000,
      temperature = 4.0,
      alpha = 0.5,
      method = "ssh",
    } = body;

    if (!teacher || !student) {
      return NextResponse.json({ error: "teacher and student are required" }, { status: 400 });
    }

    if (!SSH_PRIVATE_KEY) {
      return NextResponse.json({
        error: "SSH key not configured. Set BENCHMARK_SSH_KEY in Vercel environment variables.",
      }, { status: 503 });
    }

    const jobId = `distill_${Date.now()}`;
    const cacheDir = `~/distill_cache_${jobId}`;
    const outputDir = `~/distill_out_${jobId}`;

    // Build the full pipeline as a single script
    const pipelineScript = [
      "#!/bin/bash",
      "set -e",
      "export PATH=$HOME/.local/bin:$PATH",
      "",
      `echo "DISTILL_STATUS: caching"`,
      `python3 ~/disagg-bench/cluster/cache_teacher_logits.py \\`,
      `  --teacher "${teacher}" \\`,
      `  --student "${student}" \\`,
      `  --dataset "${dataset}" \\`,
      `  --num-samples ${numSamples} \\`,
      `  --seq-length 256 \\`,
      `  --temperature ${temperature} \\`,
      `  --alpha ${alpha} \\`,
      `  --output-dir ${cacheDir} \\`,
      `  --quantization int8`,
      "",
      `echo "DISTILL_STATUS: training"`,
      `python3 ~/disagg-bench/cluster/train_student_distill.py \\`,
      `  --student "${student}" \\`,
      `  --cache-dir ${cacheDir} \\`,
      `  --output-dir ${outputDir} \\`,
      `  --temperature ${temperature} \\`,
      `  --alpha ${alpha} \\`,
      `  --epochs 3 \\`,
      `  --batch-size 8 \\`,
      `  --lora`,
      "",
      `CKPT=$(ls -d ${outputDir}/checkpoint_epoch* | tail -1)`,
      `echo "DISTILL_STATUS: evaluating"`,
      `python3 ~/disagg-bench/cluster/evaluate_distill.py \\`,
      `  --teacher "${teacher}" \\`,
      `  --student "$CKPT" \\`,
      `  --lora-base "${student}" \\`,
      `  --dataset "${dataset}" \\`,
      `  --num-samples 500 \\`,
      `  --output ${outputDir}/eval_summary.json`,
      "",
      `echo "DISTILL_STATUS: complete"`,
      `cat ${outputDir}/eval_summary.json`,
    ].join("\n");

    const scriptName = `distill_${jobId}.sh`;

    if (method === "tinker") {
      // For Tinker, we'd run a different script. For now, return an estimate.
      return NextResponse.json({
        error: "Tinker API integration requires TINKER_API_KEY. Set it in Vercel env vars and install tinker-api on cluster.",
      }, { status: 501 });
    }

    // Write and submit the script
    await sshExec(`cat > /tmp/${scriptName} << 'ENDSCRIPT'\n${pipelineScript}\nENDSCRIPT`);
    await sshExec(`chmod +x /tmp/${scriptName}`);

    // Submit via SLURM
    const submitOut = await sshExec(
      `sbatch --job-name=${jobId} ` +
      `--output=$HOME/disagg-bench/logs/${jobId}_%j.out ` +
      `--error=$HOME/disagg-bench/logs/${jobId}_%j.err ` +
      `--ntasks=1 --cpus-per-task=8 --mem=128G ` +
      `--gres=gpu:1 --time=02:00:00 --partition=priority ` +
      `/tmp/${scriptName}`
    );

    const jobMatch = submitOut.match(/Submitted batch job (\d+)/);
    const slurmJobId = jobMatch ? jobMatch[1] : jobId;

    return NextResponse.json({
      status: "started",
      jobId: slurmJobId,
      outputDir,
    });

  } catch (err) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

export async function GET(req: NextRequest) {
  const jobId = req.nextUrl.searchParams.get("jobId");
  if (!jobId) {
    return NextResponse.json({ error: "jobId required" }, { status: 400 });
  }

  try {
    // Check if SLURM job is still running
    const squeueOut = await sshExec(
      `squeue -j ${jobId} --noheader 2>/dev/null || true`,
      30000,
    );

    if (squeueOut.trim().length > 0) {
      // Check the log file for progress
      const logContent = await sshExec(
        `tail -5 $HOME/disagg-bench/logs/*${jobId}*.out 2>/dev/null || echo "waiting"`,
        10000,
      );

      let stage = "running";
      if (logContent.includes("DISTILL_STATUS: caching")) stage = "caching";
      if (logContent.includes("DISTILL_STATUS: training")) stage = "training";
      if (logContent.includes("DISTILL_STATUS: evaluating")) stage = "evaluating";

      return NextResponse.json({ status: "running", stage, jobId });
    }

    // Job finished — find the eval summary
    const evalFiles = await sshExec(
      `ls -t ~/distill_out_*/eval_summary.json 2>/dev/null | head -1`,
      10000,
    );

    if (!evalFiles) {
      return NextResponse.json({
        status: "failed",
        error: "No evaluation results found. Check cluster logs.",
      });
    }

    const evalJson = await sshExec(`cat ${evalFiles}`, 10000);
    const evalResult = JSON.parse(evalJson);

    // Also get the distill summary for timing info
    const distillDir = evalFiles.replace("/eval_summary.json", "");
    let distillTime = 0;
    let distillCost = 0;
    try {
      const distillJson = await sshExec(`cat ${distillDir}/distill_summary.json`, 10000);
      const distillSummary = JSON.parse(distillJson);
      const totalSeconds = (distillSummary.metrics || []).reduce(
        (acc: number, m: { time_s: number }) => acc + m.time_s, 0
      );
      distillTime = totalSeconds / 60;
      distillCost = (distillTime / 60) * 45;
    } catch {
      // If distill summary is missing, estimate from eval times
      distillTime = (evalResult.teacher_eval_time_s + evalResult.student_eval_time_s) / 60;
      distillCost = (distillTime / 60) * 45;
    }

    return NextResponse.json({
      status: "complete",
      result: {
        teacher_perplexity: evalResult.teacher_perplexity,
        student_perplexity: evalResult.student_perplexity,
        token_agreement: evalResult.token_agreement,
        retention_pct: evalResult.retention_pct,
        distill_time_min: Math.round(distillTime * 10) / 10,
        distill_cost: Math.round(distillCost * 100) / 100,
      },
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
