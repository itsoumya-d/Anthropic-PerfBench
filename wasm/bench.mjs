import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const quick = process.argv.includes("--quick");
const rows = quick ? 1024 : 4096;
const cols = rows;
const group = 32;
const reps = quick ? 3 : 5;
const iters = quick ? 2 : 3;

const { instance } = await WebAssembly.instantiate(
  fs.readFileSync(
    path.join(here, "target/wasm32-unknown-unknown/release/perfbench_wasm.wasm"),
  ),
  {},
);
const e = instance.exports;

function allocBytes(length) {
  const ptr = e.pb_alloc(Math.max(1, length));
  if (!ptr) throw new Error(`pb_alloc failed for ${length} bytes`);
  return ptr;
}

class XorShift32 {
  constructor(seed) {
    this.state = seed >>> 0 || 1;
  }
  next() {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }
  unit() {
    return this.next() / 0xffffffff;
  }
}

const rng = new XorShift32(0x9e3779b9);
const f32Weights = new Float32Array(rows * cols);
for (let i = 0; i < f32Weights.length; i++) {
  f32Weights[i] = rng.unit() * 2 - 1;
}
const int4Weights = new Uint8Array((rows * cols) / 2);
for (let i = 0; i < int4Weights.length; i++) int4Weights[i] = rng.next() & 0xff;
const scales = new Float32Array((rows * cols) / group);
for (let i = 0; i < scales.length; i++) scales[i] = 0.001 + rng.unit() * 0.01;
const x = new Float32Array(cols);
for (let i = 0; i < x.length; i++) x[i] = rng.unit() * 2 - 1;

const wfPtr = allocBytes(f32Weights.byteLength);
const wiPtr = allocBytes(int4Weights.byteLength);
const sPtr = allocBytes(scales.byteLength);
const xPtr = allocBytes(x.byteLength);
const yPtr = allocBytes(rows * 4);
const yCheckPtr = allocBytes(rows * 4);

new Float32Array(e.memory.buffer, wfPtr, f32Weights.length).set(f32Weights);
new Uint8Array(e.memory.buffer, wiPtr, int4Weights.length).set(int4Weights);
new Float32Array(e.memory.buffer, sPtr, scales.length).set(scales);
new Float32Array(e.memory.buffer, xPtr, x.length).set(x);

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function measure(fn) {
  fn();
  const samples = [];
  for (let r = 0; r < reps; r++) {
    const start = performance.now();
    for (let i = 0; i < iters; i++) fn();
    samples.push((performance.now() - start) / iters);
  }
  return { best: Math.min(...samples), median: median(samples), samples };
}

const variants = [
  {
    name: "f32 scalar",
    bytesPerWeight: 4,
    run: () => e.pb_gemv_f32(wfPtr, xPtr, yPtr, rows, cols),
  },
  {
    name: "int4 scalar",
    bytesPerWeight: 0.625,
    run: () => e.pb_int4_gemv_scalar(wiPtr, sPtr, xPtr, yPtr, rows, cols, group),
  },
  {
    name: "int4 simd",
    bytesPerWeight: 0.625,
    run: () => e.pb_int4_gemv_simd(wiPtr, sPtr, xPtr, yPtr, rows, cols, group),
  },
];

const results = [];
for (const variant of variants) {
  const timing = measure(variant.run);
  const ms = timing.best;
  const gflops = (2 * rows * cols) / (ms * 1e6);
  const gbs = (rows * cols * variant.bytesPerWeight) / (ms * 1e6);
  results.push({
    name: variant.name,
    rows,
    cols,
    group,
    ms,
    median_ms: timing.median,
    gflops,
    gbs,
    samples_ms: timing.samples,
  });
}

e.pb_int4_gemv_scalar(wiPtr, sPtr, xPtr, yPtr, rows, cols, group);
e.pb_int4_gemv_simd(wiPtr, sPtr, xPtr, yCheckPtr, rows, cols, group);
const scalarOut = new Float32Array(e.memory.buffer, yPtr, rows);
const simdOut = new Float32Array(e.memory.buffer, yCheckPtr, rows);
for (let r = 0; r < rows; r++) {
  const error = Math.abs(scalarOut[r] - simdOut[r]);
  if (error > 2e-3 * (1 + Math.abs(scalarOut[r]))) {
    throw new Error(`SIMD/scalar divergence at row ${r}: ${scalarOut[r]} vs ${simdOut[r]}`);
  }
}

const scalar = results.find((r) => r.name === "int4 scalar");
const simd = results.find((r) => r.name === "int4 simd");
const report = {
  generated_at: new Date().toISOString(),
  quick,
  runtime: {
    node: process.version,
    arch: process.arch,
    cpu: os.cpus()[0]?.model ?? "unknown",
    platform: process.platform,
  },
  shape: { rows, cols, group, reps, iters },
  results,
  int4_simd_speedup_over_scalar: scalar.ms / simd.ms,
};

fs.mkdirSync(path.join(here, "results"), { recursive: true });
fs.writeFileSync(
  path.join(here, "results/bench.json"),
  JSON.stringify(report, null, 2) + "\n",
);

console.log(`WASM INT4 GEMV — ${rows}x${cols}, group ${group}`);
console.log(`runtime: ${report.runtime.cpu} / Node ${report.runtime.node}`);
console.log("variant        best ms   median ms   GFLOP/s   GB/s");
for (const row of results) {
  console.log(
    `${row.name.padEnd(14)} ${row.ms.toFixed(3).padStart(8)}  ${row.median_ms
      .toFixed(3)
      .padStart(9)}  ${row.gflops.toFixed(2).padStart(8)}  ${row.gbs
      .toFixed(2)
      .padStart(6)}`,
  );
}
console.log(
  `int4 SIMD speedup over scalar: ${report.int4_simd_speedup_over_scalar.toFixed(2)}x`,
);
console.log("wrote wasm/results/bench.json");
