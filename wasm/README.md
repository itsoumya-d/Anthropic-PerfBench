# WebAssembly SIMD INT4 GEMV

A dependency-free `wasm32-unknown-unknown` target for the group-wise INT4 GEMV
kernel in this repository. The module is built from Rust with
`-C target-feature=+simd128` and loaded directly by Node's
`WebAssembly.instantiate`; there is no wasm-bindgen, emscripten, or npm runtime.

## ABI

| Export | Signature | Purpose |
|---|---|---|
| `pb_alloc` | `(size: usize) -> *mut u8` | Allocate a 64-byte-aligned buffer |
| `pb_dealloc` | `(ptr, size)` | Release a buffer |
| `pb_gemv_f32` | `(w, x, y, rows, cols)` | Dense f32 baseline |
| `pb_int4_gemv_scalar` | `(w, scales, x, y, rows, cols, group)` | Scalar reference |
| `pb_int4_gemv_simd` | `(w, scales, x, y, rows, cols, group)` | SIMD128 fast path |

Memory is the module's default linear memory; JavaScript owns the pointers and
must call `pb_dealloc` with the same size it passed to `pb_alloc`.

## Packing contract

This matches `benchmarks/kernels.c` exactly:

- one row is `cols / 2` bytes, row-major;
- the low nibble is the even column, the high nibble is the odd column;
- nibble values are signed 4-bit two's complement (`0..7`, `-8..-1`);
- one f32 scale per `group` columns per row, row-major;
- the reference benchmark uses `group = 32`, `cols` a multiple of 32.

The SIMD path is selected only when `group == 32` and `cols % 32 == 0`.
Everything else falls back to the scalar loop rather than risking a wrong
layout.

## SIMD path

1. Load 16 bytes with `v128_load`.
2. Split with `v128_and(bytes, 0x0f)` and `u8x16_shr(bytes, 4)`.
3. Sign-extend both nibble vectors with `((q ^ 8) - 8)`.
4. Interleave low/high nibbles with two `i8x16_shuffle` ops (32 values).
5. Widen int8 → int16 → int32 → float32.
6. Multiply by four f32x4 activation vectors and accumulate.
7. Horizontal-sum the f32x4 accumulator, apply the group scale, add to the row.

The scalar Rust loop exists as the correctness oracle and as the fallback for
non-wasm builds. In native `cargo test`, `pb_int4_gemv_simd` deliberately calls
the scalar path, so SIMD parity is checked in the wasm runtime instead.

## Verification

```bash
cargo test --manifest-path wasm/Cargo.toml
cargo build --release --manifest-path wasm/Cargo.toml --target wasm32-unknown-unknown
python3 wasm/gen_fixtures.py
node wasm/parity.mjs
node wasm/bench.mjs
```

`gen_fixtures.py` is an independent float64 implementation of the packing
contract. `parity.mjs` checks the scalar and SIMD paths against that fixture and
also asserts that the activation buffer is not mutated. The committed fixture
is 16×256×32; the maximum absolute error is 1.5e-7.

## Measured results

Apple M2, Node v26.3.0, 4096×4096, group 32, best of 5 repetitions:

| Variant | ms | GFLOP/s | GB/s |
|---|---:|---:|---:|
| f32 scalar | 16.040 | 2.09 | 4.18 |
| INT4 scalar | 16.766 | 2.00 | 0.63 |
| INT4 SIMD | 2.283 | 14.70 | 4.59 |

SIMD speedup over the scalar INT4 path: 7.34×. Repeated runs varied from 11.0
to 14.7 GFLOP/s because the MacBook Air is fanless; `bench.mjs` reports both the
best and the median sample so the variance is visible. See
`results/bench.json` for the full sample set.

## Boundaries

- Single-threaded wasm; no shared memory, atomics, or worker pool.
- One row at a time; no multi-row register blocking or cache tiling.
- No SIMD f32 path; the f32 baseline is the scalar reference.
- No WebGPU comparison. The WGSL shaders remain unexecuted; no WebGPU number is
  claimed anywhere in this repository.
- The module is a benchmark/reference target, not a production inference
  runtime.
