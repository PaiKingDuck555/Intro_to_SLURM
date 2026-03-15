# What Went Wrong (Other Agent Run) + App Map for Judges

## What went wrong in the other agent’s run

### 1. **Git identity**

- The subagent tried to set **global** git config:  
  `git config --global user.email "kamrandarugar@example.com"` and a name.
- You correctly said that was wrong (wrong email, and changing global identity without asking is bad).
- **Better approach:** Use repo-only identity so the hackathon commit doesn’t touch your global git:  
  `git config user.email "you@real-email.com"` and `git config user.name "Your Name"` (no `--global`).  
  Someone later used `hackathon@disagg-bench.dev` / `Hackathon Dev` for the repo only, which is fine.

### 2. **Plan only partially executed**

- The **distillation integration plan** has 10 tasks. Only **Task 1** was fully done:
  - ✅ **Task 1:** `distill_config.py` + tests (committed).
- **Tasks 2–10 were not implemented** in that run:
  - No `cache_teacher_logits.py`, `train_student_distill.py`, SLURM scripts, integration test.
  - No Vitest, `distillation.ts`, `DistillationPlanner` component, or wiring in `page.tsx`.
  - No distillation docs in README.
- So the run stopped after Task 1 (and the spec review). The “dispatching implementers one at a time” for Tasks 2–10 either didn’t run or didn’t complete.

### 3. **API error**

- **“API Error: Unable to connect to API (ConnectionRefused)”** at the start usually means:
  - A tool or MCP server (e.g. Tinker, or another API the agent expected) wasn’t running or reachable.
  - The agent continued with clone/plan/review/execute, but any step that depended on that API would fail.

### 4. **OpenAI key**

- You pasted an OpenAI key; the agent correctly said to **revoke it** and that **this project doesn’t need it** (it uses HuggingFace OPT models on GPUs via SLURM, no OpenAI calls).  
  So that part was handled correctly; the only fix is to rotate the key if it was ever shared.

---

## What’s actually in the repo right now

| Area | Status |
|------|--------|
| **Monte Carlo Pi (SLURM intro)** | Done — CPU/GPU scripts, SLURM, combine_results |
| **disagg-bench cluster** | Benchmark + optimizer + SLURM; **distillation:** only `distill_config.py` + tests |
| **disagg-bench UI (InfraBench)** | Model selector, constraints, recommendation, Pareto chart, Fine-Tune Planner — **no Distillation Planner** |
| **Distillation plan** | Written and reviewed in `docs/superpowers/plans/2026-03-15-distillation-integration.md` — **not implemented** beyond Task 1 |

---

## App map for judges (non-technical)

**One-line pitch:**  
“We help you choose and size GPU infrastructure for large language models (LLMs), and we show how to run that workload on a shared cluster (SLURM) — from a simple Pi demo to benchmarking real models and planning distillation.”

### 1. **Monte Carlo Pi (intro to SLURM)**

- **What it is:** A small demo that estimates π by random sampling (CPU and GPU).
- **Why it’s here:** To show how to run **batch jobs on a cluster**: you submit a “job” (e.g. 10 tasks), the scheduler (SLURM) runs them on nodes, and you combine results. Same idea later for running 10 “cache teacher logits” tasks in parallel.
- **Flow:**  
  Scripts (`monte_carlo_pi.py`, `monte_carlo_pi_gpu.py`) → SLURM submits them (`submit_pi.slurm`, `submit_pi_gpu.slurm`) → cluster runs them → `combine_results.py` merges outputs.

### 2. **disagg-bench (LLM benchmarking + optimizer)**

- **What it is:** A backend that **benchmarks** LLM inference (throughput, latency, VRAM) for different model sizes and quantization (FP16, INT8, INT4), and an **optimizer** that recommends GPU setups (e.g. “use 2× B200 for prefill, 4× for decode”) to meet cost/latency goals.
- **Why it matters:** Running big models is expensive; you need real numbers and a way to compare “serve the big model” vs “quantize it” vs “distill it to a smaller model.” The benchmark data and optimizer feed that comparison.
- **Flow:**  
  Python scripts run on the cluster (SLURM) and write benchmark JSON → optimizer reads that + a GPU catalog and user constraints → recommends architectures and costs.

### 3. **InfraBench (web UI)**

- **What it is:** A Next.js app where you pick a model, set constraints (latency, budget, users), and see:
  - **Recommendation:** e.g. “Use 4× B200, INT4, $X/hr.”
  - **Pareto chart:** trade-offs between cost and latency.
  - **Fine-Tuning Planner:** rough cost/setup for LoRA/QLoRA/full fine-tuning.
- **Why it matters:** Judges can **see** the value without touching SLURM or Python; the UI answers “what GPUs do I need?” and “what does fine-tuning cost?” in one place.

### 4. **Distillation (planned, mostly not built yet)**

- **What it is:** “Knowledge distillation” = train a **small** model (student) to mimic a **large** model (teacher) so you get similar quality with cheaper inference.
- **What we planned:**  
  - **Backend:** Run the teacher once on data, save its “soft” predictions to disk (offline logit cache); then train the student from those files so the teacher never needs to sit in GPU memory during training. Run both stages via SLURM (many cache tasks in parallel, then one training job).  
  - **Frontend:** A “Distillation Planner” tab that compares “distill 66B → 1.3B” vs “just quantize 66B to INT4” in terms of cost and estimated accuracy.
- **What’s done:** Only the **config** for that pipeline (`distill_config.py` + tests). The rest (cache script, training script, SLURM jobs, UI tab) is in the plan but not implemented.

### 5. **How the pieces fit together**

```
User opens InfraBench UI
    → Picks model + constraints
    → Sees: recommended GPUs, Pareto curve, fine-tune options
    → (When built) Distillation Planner: “Distill or quantize?” + cost/accuracy

Behind the UI:
    → Optimizer uses benchmark JSON (from cluster runs) + GPU catalog
    → Benchmark JSON is produced by running benchmark_inference.py on the cluster via SLURM

Distillation (when implemented):
    → SLURM runs cache_teacher_logits (many shards in parallel)
    → Then SLURM runs train_student_distill once
    → UI shows estimated cost and accuracy for “distill” vs “quantize”
```

---

## How to demo to judges

1. **Start with the UI:** Open InfraBench, change model and constraints, show the recommendation and Pareto chart. Say: “This uses real benchmark data from our cluster to tell you how many GPUs you need and at what cost.”
2. **Then the cluster story:** “The numbers come from running standard LLM inference benchmarks on FluidStack via SLURM — the same way we run the Pi demo: submit jobs, get results, aggregate.”
3. **Then distillation:** “We’re adding distillation: train a small model to mimic the big one so you can serve cheaply. The plan is to cache the big model’s predictions once on the cluster, then train the small model from that cache so we don’t need both in GPU memory at once. Only the config and tests are in the repo so far; the full pipeline and UI tab are in the plan and can be implemented next.”

That way judges see a working product (Monte Carlo + InfraBench + optimizer), understand where the data comes from (SLURM + benchmarks), and get a clear story for what’s next (distillation pipeline + planner).
