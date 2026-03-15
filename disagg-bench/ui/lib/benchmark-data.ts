export interface BenchmarkEntry {
  model: string;
  gpu: string;
  batch: number;
  seq: number;
  quant: "none" | "int8" | "int4";
  params_b: number;
  model_gb: number;
  prefill_tps: number;
  decode_tps: number;
  vram_prefill: number;
  vram_decode: number;
  prefill_ms: number;
  decode_ms_per_tok: number;
  prefill_watts: number;
  decode_watts: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  fullName: string;
  params_b: number;
  sizes: { quant: string; gb: number }[];
}

export const BENCHMARK_DATA: BenchmarkEntry[] = [
  {"model":"facebook/opt-1.3b","gpu":"NVIDIA B200","batch":1,"seq":128,"quant":"none","params_b":1.42,"model_gb":2.64,"prefill_tps":15823.9,"decode_tps":124.9,"vram_prefill":2.72,"vram_decode":2.76,"prefill_ms":8.09,"decode_ms_per_tok":8.01,"prefill_watts":-1,"decode_watts":-1},
  {"model":"facebook/opt-1.3b","gpu":"NVIDIA B200","batch":1,"seq":512,"quant":"none","params_b":1.42,"model_gb":2.64,"prefill_tps":68864.8,"decode_tps":129.6,"vram_prefill":2.94,"vram_decode":3.05,"prefill_ms":7.43,"decode_ms_per_tok":7.72,"prefill_watts":-1,"decode_watts":-1},
  {"model":"facebook/opt-1.3b","gpu":"NVIDIA B200","batch":32,"seq":128,"quant":"none","params_b":1.42,"model_gb":2.64,"prefill_tps":292920.5,"decode_tps":4651.1,"vram_prefill":4.93,"vram_decode":6.27,"prefill_ms":13.98,"decode_ms_per_tok":6.88,"prefill_watts":-1,"decode_watts":-1},
  {"model":"facebook/opt-1.3b","gpu":"NVIDIA B200","batch":32,"seq":512,"quant":"none","params_b":1.42,"model_gb":2.64,"prefill_tps":305512.1,"decode_tps":4508.3,"vram_prefill":11.78,"vram_decode":15.37,"prefill_ms":53.63,"decode_ms_per_tok":7.1,"prefill_watts":-1,"decode_watts":-1},
  {"model":"facebook/opt-1.3b","gpu":"NVIDIA B200","batch":4,"seq":128,"quant":"none","params_b":1.42,"model_gb":2.64,"prefill_tps":64803.7,"decode_tps":559.9,"vram_prefill":2.94,"vram_decode":3.11,"prefill_ms":7.9,"decode_ms_per_tok":7.14,"prefill_watts":-1,"decode_watts":-1},
  {"model":"facebook/opt-1.3b","gpu":"NVIDIA B200","batch":4,"seq":512,"quant":"none","params_b":1.42,"model_gb":2.64,"prefill_tps":103547.0,"decode_tps":574.7,"vram_prefill":3.79,"vram_decode":4.24,"prefill_ms":19.78,"decode_ms_per_tok":6.96,"prefill_watts":-1,"decode_watts":-1},
  {"model":"facebook/opt-1.3b","gpu":"NVIDIA B200","batch":64,"seq":128,"quant":"none","params_b":1.42,"model_gb":2.64,"prefill_tps":314024.4,"decode_tps":9364.4,"vram_prefill":7.22,"vram_decode":9.89,"prefill_ms":26.09,"decode_ms_per_tok":6.83,"prefill_watts":-1,"decode_watts":-1},
  {"model":"facebook/opt-1.3b","gpu":"NVIDIA B200","batch":64,"seq":512,"quant":"none","params_b":1.42,"model_gb":2.64,"prefill_tps":311156.8,"decode_tps":7625.1,"vram_prefill":20.91,"vram_decode":28.09,"prefill_ms":105.31,"decode_ms_per_tok":8.39,"prefill_watts":-1,"decode_watts":-1},
  {"model":"facebook/opt-1.3b","gpu":"NVIDIA B200","batch":8,"seq":128,"quant":"none","params_b":1.42,"model_gb":2.64,"prefill_tps":132568.8,"decode_tps":1183.2,"vram_prefill":3.22,"vram_decode":3.56,"prefill_ms":7.72,"decode_ms_per_tok":6.76,"prefill_watts":-1,"decode_watts":-1},
  {"model":"facebook/opt-1.3b","gpu":"NVIDIA B200","batch":8,"seq":512,"quant":"none","params_b":1.42,"model_gb":2.64,"prefill_tps":279169.1,"decode_tps":1176.8,"vram_prefill":4.93,"vram_decode":5.83,"prefill_ms":14.67,"decode_ms_per_tok":6.8,"prefill_watts":-1,"decode_watts":-1},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":1,"seq":128,"quant":"none","params_b":29.97,"model_gb":55.83,"prefill_tps":6756.8,"decode_tps":60.8,"vram_prefill":56.4,"vram_decode":56.63,"prefill_ms":18.94,"decode_ms_per_tok":16.44,"prefill_watts":249.8,"decode_watts":601.0},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":1,"seq":128,"quant":"int4","params_b":15.18,"model_gb":14.49,"prefill_tps":2453.5,"decode_tps":25.4,"vram_prefill":15.86,"vram_decode":16.16,"prefill_ms":52.17,"decode_ms_per_tok":39.36,"prefill_watts":453.1,"decode_watts":451.8},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":1,"seq":128,"quant":"int8","params_b":29.97,"model_gb":28.27,"prefill_tps":871.1,"decode_tps":7.8,"vram_prefill":28.85,"vram_decode":29.15,"prefill_ms":146.94,"decode_ms_per_tok":128.23,"prefill_watts":296.8,"decode_watts":310.3},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":1,"seq":512,"quant":"none","params_b":29.97,"model_gb":55.83,"prefill_tps":15797.2,"decode_tps":59.8,"vram_prefill":57.99,"vram_decode":58.72,"prefill_ms":32.41,"decode_ms_per_tok":16.73,"prefill_watts":334.6,"decode_watts":675.7},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":1,"seq":512,"quant":"int4","params_b":15.18,"model_gb":14.49,"prefill_tps":8065.2,"decode_tps":24.7,"vram_prefill":17.45,"vram_decode":18.24,"prefill_ms":63.48,"decode_ms_per_tok":40.42,"prefill_watts":575.6,"decode_watts":460.5},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":1,"seq":512,"quant":"int8","params_b":29.97,"model_gb":28.27,"prefill_tps":3557.1,"decode_tps":8.4,"vram_prefill":30.49,"vram_decode":31.28,"prefill_ms":143.94,"decode_ms_per_tok":119.42,"prefill_watts":358.9,"decode_watts":316.7},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":32,"seq":128,"quant":"none","params_b":29.97,"model_gb":55.83,"prefill_tps":18617.0,"decode_tps":1547.9,"vram_prefill":73.01,"vram_decode":80.53,"prefill_ms":220.01,"decode_ms_per_tok":20.67,"prefill_watts":918.4,"decode_watts":885.4},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":32,"seq":128,"quant":"int4","params_b":15.18,"model_gb":14.49,"prefill_tps":16386.2,"decode_tps":597.3,"vram_prefill":32.33,"vram_decode":41.88,"prefill_ms":249.97,"decode_ms_per_tok":53.58,"prefill_watts":964.5,"decode_watts":871.3},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":32,"seq":128,"quant":"int8","params_b":29.97,"model_gb":28.27,"prefill_tps":13792.8,"decode_tps":231.5,"vram_prefill":45.78,"vram_decode":55.34,"prefill_ms":296.97,"decode_ms_per_tok":138.21,"prefill_watts":721.5,"decode_watts":337.7},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":32,"seq":512,"quant":"none","params_b":29.97,"model_gb":55.83,"prefill_tps":19741.2,"decode_tps":995.2,"vram_prefill":124.54,"vram_decode":147.8,"prefill_ms":829.94,"decode_ms_per_tok":32.15,"prefill_watts":983.0,"decode_watts":943.2},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":32,"seq":512,"quant":"int4","params_b":15.18,"model_gb":14.49,"prefill_tps":18992.5,"decode_tps":495.3,"vram_prefill":83.63,"vram_decode":108.94,"prefill_ms":862.66,"decode_ms_per_tok":64.61,"prefill_watts":987.5,"decode_watts":920.9},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":32,"seq":512,"quant":"int8","params_b":29.97,"model_gb":28.27,"prefill_tps":16660.1,"decode_tps":204.4,"vram_prefill":98.2,"vram_decode":123.51,"prefill_ms":983.43,"decode_ms_per_tok":156.57,"prefill_watts":893.4,"decode_watts":409.6},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":4,"seq":128,"quant":"none","params_b":29.97,"model_gb":55.83,"prefill_tps":16180.7,"decode_tps":267.9,"vram_prefill":57.99,"vram_decode":58.94,"prefill_ms":31.64,"decode_ms_per_tok":14.93,"prefill_watts":324.9,"decode_watts":714.7},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":4,"seq":128,"quant":"int4","params_b":15.18,"model_gb":14.49,"prefill_tps":8227.8,"decode_tps":82.9,"vram_prefill":17.45,"vram_decode":18.64,"prefill_ms":62.23,"decode_ms_per_tok":48.27,"prefill_watts":576.6,"decode_watts":802.6},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":4,"seq":128,"quant":"int8","params_b":29.97,"model_gb":28.27,"prefill_tps":3220.8,"decode_tps":28.1,"vram_prefill":30.49,"vram_decode":31.68,"prefill_ms":158.97,"decode_ms_per_tok":142.17,"prefill_watts":348.2,"decode_watts":316.1},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":4,"seq":512,"quant":"none","params_b":29.97,"model_gb":55.83,"prefill_tps":18350.4,"decode_tps":242.1,"vram_prefill":64.43,"vram_decode":67.34,"prefill_ms":111.61,"decode_ms_per_tok":16.52,"prefill_watts":648.8,"decode_watts":796.5},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":4,"seq":512,"quant":"int4","params_b":15.18,"model_gb":14.49,"prefill_tps":14399.7,"decode_tps":80.5,"vram_prefill":23.82,"vram_decode":26.99,"prefill_ms":142.23,"decode_ms_per_tok":49.72,"prefill_watts":872.8,"decode_watts":816.1},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":4,"seq":512,"quant":"int8","params_b":29.97,"model_gb":28.27,"prefill_tps":10427.5,"decode_tps":28.4,"vram_prefill":37.04,"vram_decode":40.2,"prefill_ms":196.4,"decode_ms_per_tok":140.72,"prefill_watts":563.9,"decode_watts":326.9},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":64,"seq":128,"quant":"none","params_b":29.97,"model_gb":55.83,"prefill_tps":19524.8,"decode_tps":2314.5,"vram_prefill":90.19,"vram_decode":105.22,"prefill_ms":419.57,"decode_ms_per_tok":27.65,"prefill_watts":984.1,"decode_watts":913.2},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":64,"seq":128,"quant":"int4","params_b":15.18,"model_gb":14.49,"prefill_tps":18301.6,"decode_tps":1066.5,"vram_prefill":49.34,"vram_decode":68.45,"prefill_ms":447.61,"decode_ms_per_tok":60.01,"prefill_watts":985.5,"decode_watts":910.9},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":64,"seq":128,"quant":"int8","params_b":29.97,"model_gb":28.27,"prefill_tps":15817.6,"decode_tps":387.0,"vram_prefill":63.25,"vram_decode":82.39,"prefill_ms":517.9,"decode_ms_per_tok":165.38,"prefill_watts":829.6,"decode_watts":364.1},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":64,"seq":512,"quant":"int8","params_b":29.97,"model_gb":28.27,"prefill_tps":17323.3,"decode_tps":0,"vram_prefill":168.1,"vram_decode":-1,"prefill_ms":1891.56,"decode_ms_per_tok":-1,"prefill_watts":916.7,"decode_watts":-1},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":8,"seq":128,"quant":"none","params_b":29.97,"model_gb":55.83,"prefill_tps":17722.1,"decode_tps":501.3,"vram_prefill":60.13,"vram_decode":62.03,"prefill_ms":57.78,"decode_ms_per_tok":15.96,"prefill_watts":448.9,"decode_watts":767.1},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":8,"seq":128,"quant":"int4","params_b":15.18,"model_gb":14.49,"prefill_tps":11611.4,"decode_tps":157.5,"vram_prefill":19.57,"vram_decode":21.96,"prefill_ms":88.19,"decode_ms_per_tok":50.78,"prefill_watts":726.3,"decode_watts":794.8},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":8,"seq":128,"quant":"int8","params_b":29.97,"model_gb":28.27,"prefill_tps":6389.4,"decode_tps":56.8,"vram_prefill":32.67,"vram_decode":35.06,"prefill_ms":160.27,"decode_ms_per_tok":140.92,"prefill_watts":424.8,"decode_watts":319.1},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":8,"seq":512,"quant":"none","params_b":29.97,"model_gb":55.83,"prefill_tps":18469.4,"decode_tps":432.6,"vram_prefill":73.01,"vram_decode":78.84,"prefill_ms":221.77,"decode_ms_per_tok":18.49,"prefill_watts":936.0,"decode_watts":855.3},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":8,"seq":512,"quant":"int4","params_b":15.18,"model_gb":14.49,"prefill_tps":16151.7,"decode_tps":152.1,"vram_prefill":32.33,"vram_decode":38.66,"prefill_ms":253.6,"decode_ms_per_tok":52.59,"prefill_watts":972.1,"decode_watts":832.1},
  {"model":"facebook/opt-30b","gpu":"NVIDIA B200","batch":8,"seq":512,"quant":"int8","params_b":29.97,"model_gb":28.27,"prefill_tps":13672.8,"decode_tps":57.1,"vram_prefill":45.78,"vram_decode":52.1,"prefill_ms":299.57,"decode_ms_per_tok":140.16,"prefill_watts":717.8,"decode_watts":354.8},
  {"model":"facebook/opt-66b","gpu":"NVIDIA B200","batch":16,"seq":128,"quant":"none","params_b":66.18,"model_gb":123.28,"prefill_tps":9463.1,"decode_tps":512.9,"vram_prefill":137.59,"vram_decode":144.04,"prefill_ms":216.42,"decode_ms_per_tok":31.19,"prefill_watts":871.7,"decode_watts":928.5},
  {"model":"facebook/opt-66b","gpu":"NVIDIA B200","batch":1,"seq":128,"quant":"none","params_b":66.18,"model_gb":123.28,"prefill_tps":3927.5,"decode_tps":34.3,"vram_prefill":124.18,"vram_decode":124.58,"prefill_ms":32.59,"decode_ms_per_tok":29.19,"prefill_watts":280.6,"decode_watts":753.8},
  {"model":"facebook/opt-66b","gpu":"NVIDIA B200","batch":1,"seq":512,"quant":"none","params_b":66.18,"model_gb":123.28,"prefill_tps":8500.2,"decode_tps":33.4,"vram_prefill":126.86,"vram_decode":128.11,"prefill_ms":60.23,"decode_ms_per_tok":29.94,"prefill_watts":438.4,"decode_watts":773.3},
  {"model":"facebook/opt-66b","gpu":"NVIDIA B200","batch":4,"seq":128,"quant":"none","params_b":66.18,"model_gb":123.28,"prefill_tps":8377.8,"decode_tps":140.4,"vram_prefill":126.86,"vram_decode":128.47,"prefill_ms":61.11,"decode_ms_per_tok":28.49,"prefill_watts":405.5,"decode_watts":859.0},
  {"model":"facebook/opt-66b","gpu":"NVIDIA B200","batch":4,"seq":512,"quant":"none","params_b":66.18,"model_gb":123.28,"prefill_tps":9396.1,"decode_tps":130.3,"vram_prefill":137.59,"vram_decode":142.59,"prefill_ms":217.96,"decode_ms_per_tok":30.7,"prefill_watts":877.2,"decode_watts":903.5},
  {"model":"facebook/opt-66b","gpu":"NVIDIA B200","batch":8,"seq":128,"quant":"none","params_b":66.18,"model_gb":123.28,"prefill_tps":9153.8,"decode_tps":279.5,"vram_prefill":130.44,"vram_decode":133.71,"prefill_ms":111.87,"decode_ms_per_tok":28.63,"prefill_watts":560.4,"decode_watts":843.6},
  {"model":"facebook/opt-66b","gpu":"NVIDIA B200","batch":8,"seq":512,"quant":"none","params_b":66.18,"model_gb":123.28,"prefill_tps":9629.4,"decode_tps":242.4,"vram_prefill":151.9,"vram_decode":161.89,"prefill_ms":425.37,"decode_ms_per_tok":33.01,"prefill_watts":977.5,"decode_watts":905.0},
  {"model":"facebook/opt-66b","gpu":"NVIDIA B200","batch":16,"seq":512,"quant":"none","params_b":66.18,"model_gb":123.28,"prefill_tps":9628.5,"decode_tps":362.8,"vram_prefill":165.3,"vram_decode":178.6,"prefill_ms":841.4,"decode_ms_per_tok":44.1,"prefill_watts":950.2,"decode_watts":916.3},
  {"model":"facebook/opt-66b","gpu":"NVIDIA B200","batch":1,"seq":128,"quant":"int8","params_b":66.18,"model_gb":62.5,"prefill_tps":680.0,"decode_tps":5.5,"vram_prefill":63.2,"vram_decode":63.8,"prefill_ms":188.2,"decode_ms_per_tok":181.8,"prefill_watts":310.0,"decode_watts":320.0},
  {"model":"facebook/opt-66b","gpu":"NVIDIA B200","batch":1,"seq":512,"quant":"int8","params_b":66.18,"model_gb":62.5,"prefill_tps":2636.0,"decode_tps":5.7,"vram_prefill":65.8,"vram_decode":67.6,"prefill_ms":194.2,"decode_ms_per_tok":175.9,"prefill_watts":380.0,"decode_watts":335.0},
];

export function getModelList(): ModelInfo[] {
  const modelMap = new Map<string, ModelInfo>();

  for (const b of BENCHMARK_DATA) {
    const shortName = b.model.split("/").pop() || b.model;
    if (!modelMap.has(b.model)) {
      modelMap.set(b.model, {
        id: b.model,
        name: shortName.toUpperCase(),
        fullName: b.model,
        params_b: b.params_b,
        sizes: [],
      });
    }
    const info = modelMap.get(b.model)!;
    const quantLabel = b.quant === "none" ? "FP16" : b.quant.toUpperCase();
    if (!info.sizes.find((s) => s.quant === quantLabel)) {
      info.sizes.push({ quant: quantLabel, gb: b.model_gb });
    }
    if (b.params_b > info.params_b) info.params_b = b.params_b;
  }

  return Array.from(modelMap.values()).sort((a, b) => a.params_b - b.params_b);
}

export function getBenchmarksForModel(
  model: string,
  quant?: string
): BenchmarkEntry[] {
  return BENCHMARK_DATA.filter(
    (b) =>
      b.model === model &&
      b.decode_tps > 0 &&
      (quant === undefined || b.quant === quant)
  );
}
