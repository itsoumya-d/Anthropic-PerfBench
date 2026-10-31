#include "../benchmarks/kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned state = 12345;
static float rnd(void) {
    state = state * 1664525u + 1013904223u;
    return ((state >> 8) & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
}

static int close_enough(float a, float b, float tol) {
    return fabsf(a - b) <= tol * (1.0f + fabsf(b));
}

int main(void) {
    const int n = 4096;
    float *x = malloc(sizeof(float) * n);
    float *r = malloc(sizeof(float) * n);
    float *ref = malloc(sizeof(float) * n);
    float *out = malloc(sizeof(float) * n);
    float *rope = malloc(sizeof(float) * n);
    float *cos_t = malloc(sizeof(float) * (n / 2));
    float *sin_t = malloc(sizeof(float) * (n / 2));
    for (int i = 0; i < n; i++) { x[i] = rnd(); r[i] = rnd(); }
    for (int i = 0; i < n / 2; i++) { cos_t[i] = cosf(i * 0.01f); sin_t[i] = sinf(i * 0.01f); }

    rmsnorm_scalar(x, r, ref, n, 1e-5f);
    rmsnorm_unrolled(x, r, out, n, 1e-5f);
    for (int i = 0; i < n; i++) if (!close_enough(out[i], ref[i], 1e-4f)) { printf("rmsnorm unrolled mismatch at %d\n", i); return 1; }
    rmsnorm_neon(x, r, out, n, 1e-5f);
    for (int i = 0; i < n; i++) if (!close_enough(out[i], ref[i], 1e-4f)) { printf("rmsnorm neon mismatch at %d\n", i); return 1; }

    memcpy(ref, x, sizeof(float) * n); memcpy(out, x, sizeof(float) * n); memcpy(rope, x, sizeof(float) * n);
    rope_scalar(ref, cos_t, sin_t, n);
    rope_unrolled(out, cos_t, sin_t, n);
    for (int i = 0; i < n; i++) if (!close_enough(out[i], ref[i], 1e-4f)) { printf("rope unrolled mismatch at %d\n", i); return 1; }
    rope_neon(rope, cos_t, sin_t, n);
    for (int i = 0; i < n; i++) if (!close_enough(rope[i], ref[i], 1e-4f)) { printf("rope neon mismatch at %d\n", i); return 1; }

    const int rows = 32, cols = 256, group = 32;
    uint8_t *weights = malloc((size_t)rows * cols / 2);
    float *scales = malloc(sizeof(float) * rows * (cols / group));
    float *vec = malloc(sizeof(float) * cols);
    float *y_ref = malloc(sizeof(float) * rows);
    float *y = malloc(sizeof(float) * rows);
    for (size_t i = 0; i < (size_t)rows * cols / 2; i++) weights[i] = (uint8_t)(state >> 8);
    for (int i = 0; i < rows * (cols / group); i++) scales[i] = 0.001f + fabsf(rnd()) * 0.01f;
    for (int i = 0; i < cols; i++) vec[i] = rnd();
    int4_gemv_scalar(weights, scales, vec, y_ref, rows, cols, group);
    int4_gemv_unrolled(weights, scales, vec, y, rows, cols, group);
    for (int i = 0; i < rows; i++) if (!close_enough(y[i], y_ref[i], 1e-3f)) { printf("gemv unrolled mismatch at %d: %f vs %f\n", i, y[i], y_ref[i]); return 1; }
    memset(y, 0, sizeof(float) * rows);
    int4_gemv_neon(weights, scales, vec, y, rows, cols, group);
    for (int i = 0; i < rows; i++) if (!close_enough(y[i], y_ref[i], 2e-3f)) { printf("gemv neon mismatch at %d: %f vs %f\n", i, y[i], y_ref[i]); return 1; }

    free(x); free(r); free(ref); free(out); free(rope); free(cos_t); free(sin_t);
    free(weights); free(scales); free(vec); free(y_ref); free(y);
    printf("PASS kernels\n");
    return 0;
}
