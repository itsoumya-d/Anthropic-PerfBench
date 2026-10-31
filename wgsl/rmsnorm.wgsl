// SPDX-License-Identifier: MIT
// Three iterations: scalar, workgroup reduction, vec4/unrolled.
struct Params { n: u32, eps: f32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> residual: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn rmsnorm_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  var sum = 0.0;
  for (var j = 0u; j < params.n; j = j + 1u) {
    let v = x[j] + residual[j];
    sum = sum + v * v;
  }
  let scale = inverseSqrt(sum / f32(params.n) + params.eps);
  out[i] = (x[i] + residual[i]) * scale;
}

@compute @workgroup_size(256)
fn rmsnorm_workgroup(@builtin(local_invocation_index) tid: u32) {
  var sum = 0.0;
  for (var j = tid; j < params.n; j = j + 256u) {
    let v = x[j] + residual[j];
    sum = sum + v * v;
  }
  partial[tid] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    workgroupBarrier();
  }
  let scale = inverseSqrt(partial[0] / f32(params.n) + params.eps);
  for (var j = tid; j < params.n; j = j + 256u) {
    out[j] = (x[j] + residual[j]) * scale;
  }
}

@compute @workgroup_size(256)
fn rmsnorm_vec4(@builtin(local_invocation_index) tid: u32) {
  var sum4 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  for (var j = tid * 4u; j + 3u < params.n; j = j + 1024u) {
    let v = vec4<f32>(x[j], x[j + 1u], x[j + 2u], x[j + 3u]) +
            vec4<f32>(residual[j], residual[j + 1u], residual[j + 2u], residual[j + 3u]);
    sum4 = sum4 + v * v;
  }
  let sum = sum4.x + sum4.y + sum4.z + sum4.w;
  partial[tid] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    workgroupBarrier();
  }
  let scale = inverseSqrt(partial[0] / f32(params.n) + params.eps);
  for (var j = tid * 4u; j + 3u < params.n; j = j + 1024u) {
    let v = vec4<f32>(x[j], x[j + 1u], x[j + 2u], x[j + 3u]) +
            vec4<f32>(residual[j], residual[j + 1u], residual[j + 2u], residual[j + 3u]);
    out[j] = v.x * scale; out[j + 1u] = v.y * scale;
    out[j + 2u] = v.z * scale; out[j + 3u] = v.w * scale;
  }
}
