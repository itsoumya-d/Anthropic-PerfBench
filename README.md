# Anthropic-PerfBench

AI-resistant low-level performance engineering: three transformer kernels,
three optimization iterations, two hardware targets.

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

## Build and test

```bash
make test     # correctness: scalar vs unrolled vs NEON, within tolerance
make bench    # writes results/bench.json
python3 scripts/roofline.py  # writes results/roofline.csv and roofline.svg
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

## OCI Ampere A1

`scripts/run-perf-stat.sh` builds the benchmark and runs it under Linux `perf
stat` with cache, branch, cycle, and instruction counters. The OCI ARM run and
`perf` counters were not executed in this environment.

## Tests

`tests/test_kernels.c` checks every iteration against the scalar reference for
RMSNorm, RoPE, and INT4 GEMV (relative tolerances 1e-4 / 1e-4 / 2e-3). CI runs
the scalar path on Ubuntu x86 and the NEON path on macOS arm64.

## License

MIT. Copyright (c) 2024-2026 Soumya Debnath.
