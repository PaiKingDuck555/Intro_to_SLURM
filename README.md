# Monte Carlo Pi Estimation on FluidStack via SLURM

Estimate Pi by throwing 100,000,000 random darts at a **2π × 2π** square
containing an inscribed circle of **radius π**. The ratio of darts landing
inside the circle to the total gives **π/4**, so **Pi ≈ 4 × (hits / total)**.

## Files

| File | Purpose |
|---|---|
| `monte_carlo_pi.py` | CPU simulation — generates random points and counts hits (numpy) |
| `monte_carlo_pi_gpu.py` | GPU simulation — same math but runs on CUDA cores (CuPy), includes CPU vs GPU benchmark |
| `submit_pi.slurm` | SLURM batch script — splits work across 10 array tasks (CPU) |
| `submit_pi_gpu.slurm` | SLURM batch script — runs GPU version on a GPU node |
| `combine_results.py` | Aggregates per-task JSON results into a final Pi estimate |
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

### Submit the array job (10 tasks, 10M darts each)

```bash
sbatch submit_pi.slurm
```

### Monitor progress

```bash
squeue -u $USER          # check running tasks
tail -f logs/pi_*.out    # stream output
```

### Combine results after all tasks finish

```bash
python3 combine_results.py --results-dir results
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
