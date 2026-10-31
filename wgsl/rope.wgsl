// SPDX-License-Identifier: MIT
// Three iterations: scalar pairs, four-pair unroll, vec4 pairs.
struct Params { half: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn rope_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.half) { return; }
  let a = x[i];
  let b = x[i + params.half];
  let c = cos_table[i];
  let s = sin_table[i];
  x[i] = a * c - b * s;
  x[i + params.half] = a * s + b * c;
}

@compute @workgroup_size(256)
fn rope_unrolled(@builtin(global_invocation_id) gid: vec3<u32>) {
  let base = gid.x * 4u;
  if (base + 3u >= params.half) { return; }
  for (var k = 0u; k < 4u; k = k + 1u) {
    let i = base + k;
    let a = x[i];
    let b = x[i + params.half];
    let c = cos_table[i];
    let s = sin_table[i];
    x[i] = a * c - b * s;
    x[i + params.half] = a * s + b * c;
  }
}

@compute @workgroup_size(256)
fn rope_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
  let base = gid.x * 4u;
  if (base + 3u >= params.half) { return; }
  let a = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
  let b = vec4<f32>(x[base + params.half], x[base + params.half + 1u],
                    x[base + params.half + 2u], x[base + params.half + 3u]);
  let c = vec4<f32>(cos_table[base], cos_table[base + 1u],
                    cos_table[base + 2u], cos_table[base + 3u]);
  let s = vec4<f32>(sin_table[base], sin_table[base + 1u],
                    sin_table[base + 2u], sin_table[base + 3u]);
  let out_a = a * c - b * s;
  let out_b = a * s + b * c;
  x[base] = out_a.x; x[base + 1u] = out_a.y;
  x[base + 2u] = out_a.z; x[base + 3u] = out_a.w;
  x[base + params.half] = out_b.x; x[base + params.half + 1u] = out_b.y;
  x[base + params.half + 2u] = out_b.z; x[base + params.half + 3u] = out_b.w;
}
