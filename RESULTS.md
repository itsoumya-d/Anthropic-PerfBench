# Anthropic-PerfBench Results

## Environment

| Item | Value |
|---|---|
| Date | 2026-10-31 |
| CPU | Apple Silicon (ARMv8.2-A NEON) |
| Compiler | Apple clang, `-O3 -fno-vectorize -fno-slp-vectorize` |
| Timer | `clock_gettime(CLOCK_MONOTONIC)` |
| Copy bandwidth | 66.31 GB/s |
| RMSNorm/RoPE size | 4,194,304 elements |
| INT4 GEMV shape | 4096 × 4096, group 32 |
| WGSL execution | Not run (no WebGPU runtime) |
| WASM execution | Node v26.3.0, wasm32 SIMD128, 2026-11-04 |

## Correctness

`make test` passes for all variants:
- RMSNorm residual, scalar vs unrolled vs NEON,
- RoPE scalar vs unrolled vs NEON,
- INT4 GEMV scalar vs unrolled vs NEON, including nibble sign extension and
  group-scale application.

## Raw measurements

| Kernel | Variant | ms | GFLOP/s | GB/s | Speedup |
|---|---|---:|---:|---:|---:|
| RMSNorm+residual | scalar | 6.454 | 3.249 | 12.998 | 1.00× |
| RMSNorm+residual | unrolled | 2.488 | 8.429 | 33.716 | 2.59× |
| RMSNorm+residual | NEON | 2.087 | 10.049 | 40.195 | 3.09× |
| RoPE | scalar | 1.733 | 9.681 | 29.043 | 1.00× |
| RoPE | unrolled | 1.508 | 11.125 | 33.376 | 1.15× |
| RoPE | NEON | 0.738 | 22.733 | 68.200 | 2.35× |
| INT4 GEMV | scalar | 20.686 | 1.622 | 0.508 | 1.00× |
| INT4 GEMV | unrolled | 9.320 | 3.600 | 1.129 | 2.22× |
| INT4 GEMV | NEON | 1.748 | 19.196 | 6.017 | 11.83× |

Best time over 80 iterations for RMSNorm/RoPE and 20 for GEMV. Results vary a
few percent between runs.

## WebAssembly SIMD INT4 GEMV

Date 2026-11-04. Apple M2 (arm64), Node v26.3.0, Rust 1.99.1, wasm32 with
SIMD128. Same 4096 × 4096 shape and the same packing contract as the C harness.
Best of 5 repetitions, 3 timed iterations per repetition:

| Variant | ms | GFLOP/s | GB/s | Speedup |
|---|---:|---:|---:|---:|
| f32 scalar | 16.040 | 2.09 | 4.18 | — |
| INT4 scalar | 16.766 | 2.00 | 0.63 | 1.00× |
| INT4 SIMD | 2.283 | **14.70** | 4.59 | **7.34×** |

Repeated runs on the fanless MacBook Air produced 11.0–14.7 GFLOP/s for SIMD
(6.7×–9.5× over scalar); the table records the best run, and
`wasm/results/bench.json` retains the samples and medians. This is not a
sustained-throughput claim.

Correctness: `wasm/gen_fixtures.py` computes a 16 × 256 × 32 fixture in Python
float64; `wasm/parity.mjs` checks Rust scalar and Rust SIMD against it, checks
scalar/SIMD agreement, and checks that the activation buffer is unchanged. Max
absolute error: 1.5e-7. CI runs `cargo test` natively, builds the wasm module,
runs parity, and runs a 1024 × 1024 quick benchmark.

Roofline: 0.625 bytes per weight (0.5 packed + 0.125 scale amortized) and 2 FLOP
per weight gives AI ≈ 3.2 FLOP/byte. Against the measured 66.31 GB/s copy roof
the memory roof is ≈ 212 GFLOP/s, so the 14.70 GFLOP/s result is ≈ 7% of the
memory roof and instruction-bound, like the NEON kernel. The WASM result is
≈ 77% of the separate NEON harness (19.20 GFLOP/s); the harnesses differ, so
this is context, not a controlled cross-runtime benchmark.

Reproduce:

```bash
make wasm-test
make wasm-bench
```

## Roofline

| Kernel | AI | Memory roof | NEON attained | Utilization |
|---|---:|---:|---:|---:|
| RMSNorm+residual | 0.25 | 16.58 GFLOP/s | 10.05 | 60.6% |
| RoPE | 0.33 | 22.10 GFLOP/s | 22.73 | 102.8%* |
| INT4 GEMV | 3.19 | 211.54 GFLOP/s | 19.20 | 9.1%** |

The roofline SVG is `results/roofline.svg`.

- \* RoPE's working set is partly cache-resident, so it exceeds the DRAM-based
  copy roof. This is a measurement-model limitation, not a violation of physics.
- \** INT4 GEMV performs nibble extraction, sign extension, int→float conversion,
  and group scaling per weight. Those instructions, not DRAM bandwidth, bound the
  kernel; a memory roofline alone does not describe it. The 11.83× progression is
  still the key result.

## Iteration notes

**RMSNorm + residual**
1. Scalar two-pass: add residual, sum squares; second pass scales and writes.
2. Four independent scalar accumulators and unrolled stores; removes the serial
   dependency in the reduction.
3. NEON `float32x4_t` accumulation with `vfmaq_f32`, `vaddvq_f32` horizontal
   reduction, and vector stores. 3.09× over scalar.

**RoPE**
1. Scalar in-place rotation over pair `i` and `i + n/2`.
2. Four-pair unroll; reduces loop and table-load overhead.
3. NEON loads both halves and cos/sin as `float32x4_t`, uses multiply/add/sub,
   and stores both halves. 2.35× over scalar.

**INT4 GEMV**
1. Scalar nibble extraction with per-group scale.
2. Group-blocked, 4-way unrolled accumulation; still scalar dequantization.
3. NEON loads 16 bytes (32 nibbles), extracts low/high nibbles, sign-extends with
   XOR/subtract, interleaves with `vzip1q_s8`/`vzip2q_s8`, widens int8→int32→
   float32, and FMA-accumulates 32 weights per group. 11.83× over scalar.

## Not executed

- **OCI Ampere Altra A1** benchmarks and `perf stat` hardware counters: the
  script is `scripts/run-perf-stat.sh`; no OCI instance/credentials available.
- **WebGPU WGSL** timing via Chrome timestamp queries: shaders and an HTML
  harness are provided; no WebGPU-capable browser automation was configured.
  The WASM SIMD path *was* executed and measured; it is not a WebGPU result.
- **Roofline compute ceiling**: only the measured memory roof is plotted. A
  compute roof would require a peak-FLOPs microbenchmark or hardware counter
  calibration.

## Reproduce

```bash
cd repos/Anthropic-PerfBench
make test
make bench
python3 scripts/roofline.py
# Linux ARM with perf:
./scripts/run-perf-stat.sh
```
