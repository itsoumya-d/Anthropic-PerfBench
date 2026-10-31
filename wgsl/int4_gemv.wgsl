// SPDX-License-Identifier: MIT
// Three iterations: scalar row, workgroup reduction, word-at-a-time vec4 unpack.
struct Params { rows: u32, cols: u32, group_size: u32, groups: u32 };
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> partial: array<f32, 256>;

fn nibble(word: u32, index: u32) -> f32 {
  let q = i32((word >> (index * 4u)) & 0xFu);
  return f32(select(q, q - 16, q >= 8));
}

@compute @workgroup_size(1)
fn int4_gemv_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= params.rows) { return; }
  var sum = 0.0;
  let words_per_row = params.cols / 8u;
  for (var k = 0u; k < params.cols; k = k + 1u) {
    let word = weights[r * words_per_row + k / 8u];
    let q = nibble(word, k % 8u);
    let scale = scales[r * params.groups + k / params.group_size];
    sum = sum + q * scale * x[k];
  }
  y[r] = sum;
}

@compute @workgroup_size(256)
fn int4_gemv_workgroup(@builtin(global_invocation_id) gid: vec3<u32>,
                       @builtin(local_invocation_index) tid: u32) {
  let r = gid.x / 256u;
  if (r >= params.rows) { return; }
  var sum = 0.0;
  let words_per_row = params.cols / 8u;
  for (var k = tid; k < params.cols; k = k + 256u) {
    let word = weights[r * words_per_row + k / 8u];
    let q = nibble(word, k % 8u);
    let scale = scales[r * params.groups + k / params.group_size];
    sum = sum + q * scale * x[k];
  }
  partial[tid] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    workgroupBarrier();
  }
  if (tid == 0u) { y[r] = partial[0]; }
}

@compute @workgroup_size(256)
fn int4_gemv_vec4(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(local_invocation_index) tid: u32) {
  let r = gid.x / 256u;
  if (r >= params.rows) { return; }
  var sum4 = vec4<f32>(0.0);
  let words_per_row = params.cols / 8u;
  for (var k = tid * 4u; k + 3u < params.cols; k = k + 1024u) {
    let word = weights[r * words_per_row + k / 8u];
    let q = vec4<f32>(nibble(word, (k % 8u)), nibble(word, (k + 1u) % 8u),
                      nibble(word, (k + 2u) % 8u), nibble(word, (k + 3u) % 8u));
    let scale = scales[r * params.groups + k / params.group_size];
    let xv = vec4<f32>(x[k], x[k + 1u], x[k + 2u], x[k + 3u]);
    sum4 = sum4 + q * xv * scale;
  }
  partial[tid] = sum4.x + sum4.y + sum4.z + sum4.w;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    workgroupBarrier();
  }
  if (tid == 0u) { y[r] = partial[0]; }
}
