# Anthropic-PerfBench

AI-resistant low-level performance engineering: three transformer kernels,
three optimization iterations, three hardware targets.

Inspired by Anthropic's 3-iteration performance-engineering evaluation and
FlashAttention-2 (arXiv:2307.08691).

## Kernels

| Kernel | Iteration 1 | Iteration 2 | Iteration 3 |
|---|---|---|---|
| Fused RMSNorm + residual | two-pass scalar | 4-way unrolled scalar | NEON `float32x4_t` FMA reduction |
| Fused RoPE in-place | scalar pairs | four-pair unroll | NEON vector rotation |
| Group-wise INT4 dequant + GEMV | scalar nibble loop | group-blocked 4-way unroll | NEON nibble unpack + FMA |

Group size is 32 for INT4 GEMV; one row is one output. Weights use low-nibble
first packing, and each group has one FP32 scale.

## Targets

- **ARMv8.2-A NEON** — C intrinsics, compiled on Apple Silicon. The same source
  builds on an OCI Ampere Altra A1 with `-DUSE_NEON`.
- **WebGPU WGSL** — `wgsl/rmsnorm.wgsl`, `wgsl/rope.wgsl`,
  `wgsl/int4_gemv.wgsl`, each with three entry points (`scalar`, `workgroup`/
  `unrolled`, `vec4`). `wgsl/harness.html` initializes WebGPU and compiles the
  shaders. The WGSL path is provided but was **not executed** in this
  environment; no WebGPU numbers are claimed.
- **WebAssembly SIMD128** — a dependency-free Rust `wasm32-unknown-unknown`
  module in `wasm/` with scalar and SIMD INT4 GEMV entry points, a Python
  parity fixture, and a Node benchmark. This target **was executed** on the
  same Apple M2. See [WebAssembly SIMD](#webassembly-simd).

## Build and test

```bash
make test       # correctness: scalar vs unrolled vs NEON, within tolerance
make bench      # writes results/bench.json
python3 scripts/roofline.py  # writes results/roofline.csv and roofline.svg
make wasm-test  # Rust tests + Python/Rust/JS parity
make wasm-bench # writes wasm/results/bench.json
```

On macOS/Linux ARM the Makefile defines `USE_NEON`; on x86 the NEON entry
points fall back to the unrolled scalar path so correctness tests still run.

## Measured NEON results

Apple Silicon, clang `-O3`, 4M elements for RMSNorm/RoPE, 4096×4096 INT4 GEMV.
Measured `memcpy` bandwidth: **66.31 GB/s**.

| Kernel | Variant | ms | GFLOP/s | Speedup |
|---|---|---:|---:|---:|
| RMSNorm+residual | scalar | 6.454 | 3.25 | 1.00× |
| RMSNorm+residual | unrolled | 2.488 | 8.43 | 2.59× |
| RMSNorm+residual | NEON | 2.087 | 10.05 | **3.09×** |
| RoPE | scalar | 1.733 | 9.68 | 1.00× |
| RoPE | unrolled | 1.508 | 11.13 | 1.15× |
| RoPE | NEON | 0.738 | 22.73 | **2.35×** |
| INT4 GEMV | scalar | 20.686 | 1.62 | 1.00× |
| INT4 GEMV | unrolled | 9.320 | 3.60 | 2.22× |
| INT4 GEMV | NEON | 1.748 | 19.20 | **11.83×** |

Roofline (memory roof = 66.31 GB/s):

| Kernel | AI (FLOP/byte) | Roof GFLOP/s | NEON utilization |
|---|---:|---:|---:|
| RMSNorm+residual | 0.25 | 16.58 | 60.6% |
| RoPE | 0.33 | 22.10 | 102.8%* |
| INT4 GEMV | 3.19 | 211.54 | 9.1%** |

\* RoPE slightly exceeds the DRAM roof because the working set is partly
cache-resident; the copy bandwidth is a DRAM estimate.
\** INT4 GEMV is instruction-bound by nibble unpack/conversion, not memory-bound;
the memory roof is not its limiting ceiling.

Full data: `results/bench.json`, `results/roofline.csv`, `results/roofline.svg`.
See [RESULTS.md](RESULTS.md) for methodology and caveats.

## WebAssembly SIMD

`wasm/` builds the same group-wise INT4 GEMV contract into a self-contained
`wasm32-unknown-unknown` module. It has no wasm-bindgen or npm dependencies:
the module exports `pb_alloc`, `pb_dealloc`, `pb_gemv_f32`,
`pb_int4_gemv_scalar` and `pb_int4_gemv_simd`.

The SIMD kernel loads 16 packed bytes (32 nibbles), extracts low/high nibbles,
sign-extends them with the same XOR/subtract transform as the NEON path,
interleaves with `i8x16_shuffle`, widens int8 → int16 → int32 → float32, and
FMA-accumulates one 32-element group before applying the group scale.

Measured on the same Apple M2 (Node v26.3.0, best of 5 repetitions, 3 timed
iterations per repetition):

| Variant | Best ms | GFLOP/s | GB/s | Speedup |
|---|---:|---:|---:|---:|
| f32 scalar | 16.040 | 2.09 | 4.18 | — |
| INT4 scalar | 16.766 | 2.00 | 0.63 | 1.00× |
| INT4 SIMD | 2.283 | **14.70** | 4.59 | **7.34×** |

The 4096×4096 working set is 8 MiB of packed weights plus 512 KiB of group
scales; the f32 baseline is 64 MiB. Repeated runs on a fanless MacBook Air
ranged from 11.0 to 14.7 GFLOP/s for the SIMD path (6.7× to 9.5× over scalar),
so treat the best-of figure as a peak, not a sustained throughput guarantee.
The separate NEON harness measured 19.20 GFLOP/s for the same shape; the WASM
path reaches about 77% of that, but the two harnesses are not a controlled
cross-runtime comparison.

Cross-language parity is enforced in CI: `wasm/gen_fixtures.py` computes the
expected outputs in Python float64, and `wasm/parity.mjs` checks both the Rust
scalar and SIMD paths against it. The committed 16×256 fixture has a maximum
absolute error of **1.5e-7**.

Boundaries: the SIMD path requires `group_size == 32` and `cols % 32 == 0`;
other shapes fall back to the Rust scalar loop. There is no threading, no
multi-row blocking, and no SIMD f32 path. Roofline context: at 0.625 bytes per
weight and 2 FLOP per weight the arithmetic intensity is about 3.2 FLOP/byte;
against the measured 66.31 GB/s copy roof that is a 212 GFLOP/s memory roof, so
the 14.70 GFLOP/s result is instruction-bound at roughly 7% of the memory roof,
consistent with the NEON analysis.

```bash
make wasm-test
make wasm-bench
```

Full data: `wasm/results/bench.json`. Implementation notes:
[`wasm/README.md`](wasm/README.md).

## OCI Ampere A1

`scripts/run-perf-stat.sh` builds the benchmark and runs it under Linux `perf
stat` with cache, branch, cycle, and instruction counters. The OCI ARM run and
`perf` counters were not executed in this environment.

## Tests

`tests/test_kernels.c` checks every iteration against the scalar reference for
RMSNorm, RoPE, and INT4 GEMV (relative tolerances 1e-4 / 1e-4 / 2e-3). CI runs
the scalar path on Ubuntu x86 and the NEON path on macOS arm64. The WASM job
runs the Rust tests, the Python/Rust/JavaScript parity fixture, and a quick
benchmark on Ubuntu with Node 20.

## License

MIT. Copyright (c) 2024-2026 Soumya Debnath.
