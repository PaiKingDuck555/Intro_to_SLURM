# Monte Carlo Pi Estimation on FluidStack via SLURM

Estimate Pi by throwing random darts at a **2π × 2π** square containing an inscribed circle of **radius π**. The ratio of darts inside the circle to the total gives **π/4**, so **Pi ≈ 4 × (hits / total)**.

This repo is an introduction to SLURM and GPU HPC: you get **three ways** to run the same Monte Carlo problem—CPU array, single GPU, and **GPU distillation** (teacher + student array) with variance and confidence intervals.

## Three ways to estimate π

| Method | What it does | SLURM | Combine step |
|--------|----------------|-------|----------------|
| **CPU array** | 10 tasks × 10M darts (numpy) | `sbatch submit_pi.slurm` | `python3 combine_results.py --results-dir results` |
| **GPU single** | One big GPU run + CPU benchmark (CuPy) | `sbatch submit_pi_gpu.slurm` | — (writes `results/gpu_vs_cpu.json`) |
| **GPU distillation** | 1 teacher + 10 students (GPU); optimal use of cluster | `./run_distillation_slurm.sh` then combine | `python3 combine_results.py --results-dir results --mode distill` |

All SLURM commands are run from the **repo root** after `cd Intro_to_SLURM`.

## Quick demo (no cluster)

See the full distillation pipeline in a few seconds on your laptop (CPU fallback, small darts):

```bash
cd Intro_to_SLURM
chmod +x demo_distillation.sh run_distillation_slurm.sh
./demo_distillation.sh
```

Output: teacher + 4 students → combined π with **standard error and 95% CI** in `results/distilled_summary.json`.

## Files

| File | Purpose |
|------|--------|
| `monte_carlo_pi.py` | CPU simulation — random points and hits (numpy) |
| `monte_carlo_pi_gpu.py` | GPU simulation (CuPy), CPU vs GPU benchmark |
| `distill_pi.py` | **Distillation**: teacher + student GPU runs, variance/CI |
| `submit_pi.slurm` | SLURM array — 10 CPU tasks |
| `submit_pi_gpu.slurm` | SLURM — single GPU job (benchmark) |
| `submit_pi_teacher.slurm` | SLURM — one teacher GPU run |
| `submit_pi_students.slurm` | SLURM array — 10 student GPU runs |
| `combine_results.py` | Combines `task_*.json` (array) or teacher + `student_*.json` (`--mode distill`) |
| `demo_distillation.sh` | **Run distillation locally** (small darts, no SLURM) |
| `run_distillation_slurm.sh` | **Submit teacher → students (chained)** on SLURM |
| `requirements.txt` | Python dependencies |

## Connect to FluidStack

1. **Create a FluidStack account** at [fluidstack.io](https://www.fluidstack.io/)
   and provision a CPU (or GPU) instance.

2. **SSH into your instance:**
   ```bash
   ssh -i ~/.ssh/your_key ubuntu@<FLUIDSTACK_IP>
   ```

3. **Clone this repo on the remote machine:**
   ```bash
   git clone <YOUR_REPO_URL> ~/Intro_to_SLURM
   cd ~/Intro_to_SLURM
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Run with SLURM

From the repo root: `cd Intro_to_SLURM` (or wherever you cloned), then:

### 1. CPU array (10 tasks, 10M darts each)

```bash
sbatch submit_pi.slurm
# when done:
python3 combine_results.py --results-dir results
```

### 2. GPU single run (benchmark)

```bash
sbatch submit_pi_gpu.slurm
# writes results/gpu_vs_cpu.json (no combine step)
```

### 3. GPU distillation (teacher + students, one command)

```bash
./run_distillation_slurm.sh
# when both jobs finish:
python3 combine_results.py --results-dir results --mode distill
```

The script submits the teacher job, then the student array with `--dependency=afterok:$TEACHER_JOB` so students start only after the teacher completes.

### Monitor jobs

```bash
squeue -u $USER          # check running tasks
tail -f logs/pi_*.out    # stream output
```

### Run locally (single process, no SLURM)

```bash
python3 monte_carlo_pi.py --total-darts 100000000 --task-id 0 --num-tasks 1
```

## GPU vs CPU Comparison

On a FluidStack GPU instance, run the benchmark to see the speed difference:

```bash
pip install cupy-cuda12x
python3 monte_carlo_pi_gpu.py --total-darts 100000000
```

This runs the same 100M-dart simulation on both the GPU and CPU, then prints
the speedup. Typical results on an A100:

```
  GPU run: 100,000,000 darts
  Pi ≈ 3.141582412000    Time: 0.08s

  CPU run: 100,000,000 darts
  Pi ≈ 3.141582412000    Time: 8.50s

  SPEEDUP: GPU is 106.3x faster than CPU
```

If the GPU has limited VRAM (< 4 GB), use chunked mode:

```bash
python3 monte_carlo_pi_gpu.py --total-darts 100000000 --chunked
```

Run CPU-only (no GPU needed):

```bash
python3 monte_carlo_pi_gpu.py --total-darts 100000000 --force-cpu
```

## Distillation (GPU optimization)

**Distillation** splits work into one **teacher** run (many darts) and many **student** runs (fewer darts each, parallel). Same total darts, same optimal π estimate—but:

- **Better GPU use**: teacher runs once; students run in parallel as an array job.
- **Uncertainty quantification**: combined run reports standard error and **95% confidence interval** (true π typically lies inside it).
- **Same SLURM patterns** as the CPU array: array job + combine script, so it fits the “intro to SLURM” goal.

### One-command SLURM submission (recommended)

```bash
./run_distillation_slurm.sh
# When both jobs finish:
python3 combine_results.py --results-dir results --mode distill
```

This submits the teacher, then the student array with a dependency so students start after the teacher completes.

### Manual submission

```bash
sbatch submit_pi_teacher.slurm
sbatch submit_pi_students.slurm   # or: sbatch --dependency=afterok:<teacher_job_id> submit_pi_students.slurm
python3 combine_results.py --results-dir results --mode distill
```

Output: `results/distilled_summary.json` (π estimate, error, std error, 95% CI) and a one-line summary in the terminal.

### Low-VRAM GPUs

Use chunked mode so each run uses less GPU memory: `CHUNKED=1 sbatch submit_pi_students.slurm` (and optionally add `--chunked` when running the teacher via `distill_pi.py`).

## How It Works

```
       ┌───────────────────┐  2π
       │                   │
       │     ·  ·····  ·   │
       │   ··/────────\··  │
       │  ·/   ·  ·  · \· │
       │  |  ·    ·   · | │
       │  |·   ·(0,0)·  | │  Circle r = π
       │  |  ·  ·   ·   | │
       │  ·\  ·   ·   /·  │
       │   ··\────────/··  │
       │     ·  ·····  ·   │
       │                   │
       └───────────────────┘
              2π

  Area_circle / Area_square  =  π·r² / (2r)²  =  π / 4

  →  Pi  ≈  4  ×  (darts in circle / total darts)
```

Each SLURM array task independently generates its share of darts using a
different random seed, writes results to `results/task_<id>.json`, and
`combine_results.py` merges them into a weighted overall estimate.

---

## InfraBench — LLM GPU Infrastructure Planner

`disagg-bench/` extends this repo with a full LLM infrastructure benchmarking and
planning system for production GPU deployments.

### Components

| Directory | What it does |
|-----------|--------------|
| `disagg-bench/cluster/` | SLURM pipeline: cache teacher soft-labels + train distilled student |
| `disagg-bench/ui/` | Next.js dashboard: GPU optimizer, Pareto chart, Distillation Planner, Fine-Tuning estimator |
| `disagg-bench/server/` | Benchmark server |
| `disagg-bench/results/` | Raw benchmark JSON files |

### Knowledge Distillation Pipeline

The `disagg-bench/cluster/` pipeline solves the VRAM bottleneck for large-model
distillation using **offline logit caching**:

```
Phase 1 — Cache (parallel, SLURM array)
  Teacher model (e.g. OPT-66B INT4) runs alone per shard
  Saves softmax(logits / T) to disk as .pt files

Phase 2 — Train (single job, after Phase 1)
  Student model trains from cached tensors
  Teacher never in VRAM during training
  Loss = α·CE + (1-α)·T²·KL
```

```bash
# Submit cache array (10 shards in parallel)
CACHE_ID=$(sbatch --parsable disagg-bench/cluster/submit_cache_logits.slurm)

# Submit student training (auto-starts after all shards succeed)
sbatch --dependency=afterok:$CACHE_ID disagg-bench/cluster/submit_distill_student.slurm
```

See [`disagg-bench/cluster/README.md`](disagg-bench/cluster/README.md) for full docs.

### InfraBench UI

```bash
cd disagg-bench/ui
npm install
npm run dev        # → http://localhost:3000
npm test           # run Vitest unit tests
```

The dashboard has four sections:

1. **GPU Selector** — pick GPUs to compare
2. **Pareto Chart** — latency vs. cost scatter for all inference architectures
3. **Distillation Planner** — interactive tool: select teacher→student pair, set
   temperature (τ) and alpha (α), see estimated accuracy retention, cost savings,
   and SLURM pipeline steps
4. **Fine-Tuning Estimator** — estimate VRAM, time, and cost for full/LoRA/QLoRA
