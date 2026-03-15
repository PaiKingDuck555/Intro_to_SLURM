export interface CatalogModel {
  id: string;
  name: string;
  family: string;
  params_b: number;
  fp16_gb: number;
  hasBenchmark: boolean;
}

export const MODEL_CATALOG: CatalogModel[] = [
  // ── Benchmarked (real data) ──
  { id: "facebook/opt-1.3b", name: "OPT-1.3B", family: "OPT", params_b: 1.3, fp16_gb: 2.6, hasBenchmark: true },
  { id: "facebook/opt-30b", name: "OPT-30B", family: "OPT", params_b: 30, fp16_gb: 55.8, hasBenchmark: true },
  { id: "facebook/opt-66b", name: "OPT-66B", family: "OPT", params_b: 66, fp16_gb: 123.3, hasBenchmark: true },

  // ── Popular models (no benchmark yet) ──
  { id: "meta-llama/Llama-3.1-8B", name: "Llama 3.1 8B", family: "Llama", params_b: 8, fp16_gb: 16, hasBenchmark: false },
  { id: "meta-llama/Llama-3.1-70B", name: "Llama 3.1 70B", family: "Llama", params_b: 70, fp16_gb: 140, hasBenchmark: false },
  { id: "meta-llama/Llama-3.1-405B", name: "Llama 3.1 405B", family: "Llama", params_b: 405, fp16_gb: 810, hasBenchmark: false },
  { id: "mistralai/Mistral-7B-v0.3", name: "Mistral 7B", family: "Mistral", params_b: 7, fp16_gb: 14, hasBenchmark: false },
  { id: "mistralai/Mixtral-8x7B-v0.1", name: "Mixtral 8x7B", family: "Mistral", params_b: 47, fp16_gb: 94, hasBenchmark: false },
  { id: "mistralai/Mixtral-8x22B-v0.1", name: "Mixtral 8x22B", family: "Mistral", params_b: 141, fp16_gb: 282, hasBenchmark: false },
  { id: "google/gemma-2-9b", name: "Gemma 2 9B", family: "Gemma", params_b: 9, fp16_gb: 18, hasBenchmark: false },
  { id: "google/gemma-2-27b", name: "Gemma 2 27B", family: "Gemma", params_b: 27, fp16_gb: 54, hasBenchmark: false },
  { id: "Qwen/Qwen2.5-7B", name: "Qwen 2.5 7B", family: "Qwen", params_b: 7, fp16_gb: 14, hasBenchmark: false },
  { id: "Qwen/Qwen2.5-72B", name: "Qwen 2.5 72B", family: "Qwen", params_b: 72, fp16_gb: 144, hasBenchmark: false },
  { id: "microsoft/phi-3-mini-4k-instruct", name: "Phi-3 Mini", family: "Phi", params_b: 3.8, fp16_gb: 7.6, hasBenchmark: false },
  { id: "microsoft/phi-3-medium-4k-instruct", name: "Phi-3 Medium", family: "Phi", params_b: 14, fp16_gb: 28, hasBenchmark: false },
  { id: "deepseek-ai/DeepSeek-V2-Lite", name: "DeepSeek V2 Lite", family: "DeepSeek", params_b: 16, fp16_gb: 32, hasBenchmark: false },
  { id: "01-ai/Yi-1.5-34B", name: "Yi 1.5 34B", family: "Yi", params_b: 34, fp16_gb: 68, hasBenchmark: false },
];

export const FAMILIES = Array.from(new Set(MODEL_CATALOG.map((m) => m.family)));
