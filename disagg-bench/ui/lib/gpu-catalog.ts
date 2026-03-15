export interface GpuSpec {
  id: string;
  name: string;
  vram_gb: number;
  bandwidth_tb_s: number;
  flops_fp16_tflops: number;
  tdp_watts: number;
  price_per_hour: number;
}

export const GPU_CATALOG: Record<string, GpuSpec> = {
  B200: {
    id: "B200",
    name: "NVIDIA B200",
    vram_gb: 183,
    bandwidth_tb_s: 8.0,
    flops_fp16_tflops: 2250,
    tdp_watts: 1000,
    price_per_hour: 45.0,
  },
  H100_SXM: {
    id: "H100_SXM",
    name: "NVIDIA H100 SXM",
    vram_gb: 80,
    bandwidth_tb_s: 3.35,
    flops_fp16_tflops: 990,
    tdp_watts: 700,
    price_per_hour: 30.0,
  },
  A100_80GB: {
    id: "A100_80GB",
    name: "NVIDIA A100 80GB",
    vram_gb: 80,
    bandwidth_tb_s: 2.0,
    flops_fp16_tflops: 312,
    tdp_watts: 400,
    price_per_hour: 10.0,
  },
  A100_40GB: {
    id: "A100_40GB",
    name: "NVIDIA A100 40GB",
    vram_gb: 40,
    bandwidth_tb_s: 1.6,
    flops_fp16_tflops: 312,
    tdp_watts: 400,
    price_per_hour: 6.0,
  },
  L40S: {
    id: "L40S",
    name: "NVIDIA L40S",
    vram_gb: 48,
    bandwidth_tb_s: 0.864,
    flops_fp16_tflops: 362,
    tdp_watts: 350,
    price_per_hour: 4.0,
  },
};

export const GPU_LIST = Object.values(GPU_CATALOG);
